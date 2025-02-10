import os
import sys
import argparse
import torch
import torch.distributed as dist
import math
import time
import numpy as np
import json
import random
from sympy import *
from decimal import Decimal
from queue import Queue
from threading import Thread
from tasks import get_task_data,reset_ps_party
from itertools import combinations, permutations
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, get_random_computation_time
from collections import deque
import datetime

import threading
lock = threading.Lock()

# device = torch.device("cpu")
gr = (math.sqrt(5) + 1) / 2
bandwidth_mbps = 300
ts1_wait_time = 0
max_comm_time = 0
comm_volume = 0
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--ps_ip', default='localhost', type=str, help='ip of ps')
    parser.add_argument('--ps_port', default='8888', type=str, help='port of ps')
    parser.add_argument('--task_name', default='mnist', type=str, help='task name')
    # parser.add_argument('--use_gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--is_asyn', action='store_true', help='asynchronous training or not')
    parser.add_argument('--gpu', default=-1, type=int, help='GPU ID, 0 or 1. -1 for cpu')

    parser.add_argument('--use_reweight', action='store_true', help='reweight or not')
    parser.add_argument('--use_freezing', action='store_true', help='freezing or not')
    parser.add_argument('--warm_up_phase', default=1, type=int, help='warm up phase')
    parser.add_argument('--validation_alpha', default=5, type=int, help='validation alpha')
    parser.add_argument('--validation_period', default=40, type=int, help='validation period')    
    parser.add_argument('--sv_sampling_times', default=10, type=int, help='sv sampling times')
    parser.add_argument('--performance_threshold_rate', default=0.9, type=float, help='performance threshold')
    parser.add_argument('--exploration_weight', default=0.1, type=float, help='exploration weight')
    parser.add_argument('--freezing_threshold', default=0.015, type=float, help='freezing threshold')
    parser.add_argument('--selection_threshold_rate', default=0.75, type=float, help='selection threshold')
    parser.add_argument('--unselected_threshold', default=10, type=int, help='多少轮没被选中算“长期未选择”')
    parser.add_argument('--reweight_beta', default=5, type=float, help='权重降低到多少')

    args = parser.parse_args()
    print(args)

    task_info = json.load(open('task_info.json','r',encoding='utf-8'))[args.task_name]
    global div
    div = task_info['div']
    print(div)
    global device
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu!= -1 and torch.cuda.is_available() else 'cpu')
    print(device)
    print('reweight:', args.use_reweight)

    backend = 'gloo'
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    if sys.platform == 'linux':
        network_card_list = os.listdir('/sys/class/net/')
        if "eth0" in network_card_list:
            os.environ['TP_SOCKET_IFNAME'] = "eth0"
            os.environ['GLOO_SOCKET_IFNAME'] = "eth0"
    dist.init_process_group(backend=backend,world_size=args.world_size, rank=args.rank)

    # ntp_client = ntplib.NTPClient()
    # ntp_server = 'pool.ntp.org'
    # response = ntp_client.request(ntp_server)
    # server_time = response.tx_time
    # time.time = lambda: server_time
    # print('start time:', time.time())

    if args.rank == 0:
        run_ps(args)
    else:
        run_client(args)

def gss(f, a, b, tol=1e-5):
    """Golden-section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if f(c) < f(d):  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


def run_ps(args):
    rank = args.rank
    rank_list = [i+1 for i in range(args.world_size-1)]
    party,train_loader,test_loader,epochs,bound,lr,delta_T,CT = get_task_data(task_name=args.task_name,id=0,is_asyn=args.is_asyn,gpu=args.gpu)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 10

    select_times = 0 # 选择次数
    avg_contribution = [0] * (args.world_size - 1) # 平均贡献度
    selected_times_list = [0] * (args.world_size - 1)
    ucb_values = [0] * (args.world_size - 1)
    unselected_count_list = [0] * (args.world_size - 1)
    h_undo_list = [0] * (args.world_size - 1)
    selected_num = 10
    selected_clients = [0,1,2,3,4,5,6,7,8,9]
    #selected_clients = [1,5,9]
    party.h_weight_list = torch.zeros(args.world_size - 1)
    for client in selected_clients:
        party.h_weight_list[client] = 1
    party.h_weight_list = party.h_weight_list / sum(party.h_weight_list)
    print(party.h_weight_list)
    # select_num = 5
    # selected_clients = [0,1,2,3,4]
    # party.h_weight_list = [1/(args.world_size - 1)] * (args.world_size - 1)

    # selected_clients = [0,2,3,4,5]
    # party.h_weight_list = torch.tensor([0.2, 0, 0.2, 0.2, 0.2, 0.2])

    # selected_clients = [0,2,3,5]
    # party.h_weight_list = torch.tensor([0.25, 0, 0.25, 0.25, 0, 0.25])
    last_theta = None
    global_step = 0
    last_validation_step = 0
    running_time = 0
    select_time = 0
    recv_wait_time = 0
    t2_total = 0
    t2_first = 0
    t2_last = 0

    t2 = 0
    last_t2 = 0
    T_step = 0
    local_step = 0
    last_running_time = 0
    gap_time = 0
    gap_scale = ((np.random.rand()+0.5)**2-0.25)*10

    Q = party.n_iter
    Q_l = Q
    Q_last = Q_l
    D = bound if bound > 0 else 1
    T = train_batches * epochs
    N = T / D
    N_prime = delta_T / D
    Cl = None
    c0 = lr * (T**0.5)
    loss_l = None
    E = loss_l
    G = None
    
    shape_list = []
    predict_shape_list = []

    is_finish = False

    # h_list_queue = Queue()
    h_queue_list = [Queue() for _ in rank_list]
    ts1_queue_list = [Queue() for _ in rank_list]
    ts2_queue_list = [Queue() for _ in rank_list]
    grad_list_queue = Queue()
    predict_h_list_queue = Queue()
    # staleness_list_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',grad_list_queue,rank_list,0))
    send_thread.start()
    # pull_staleness_thread = Thread(target=process_communicate,daemon=True,args=('pull',staleness_list_queue,rank_list,4,[[] for _ in range(args.world_size-1)]))
    # pull_staleness_thread.start()

    freeze_list = [False for _ in rank_list]
    flag_queue_list = [Queue() for _ in rank_list]

    log_dir = os.path.join('summary_pic',args.task_name,time.strftime("%Y%m%d-%H%M"))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("Q_l&step", Q_l, global_step)
    writer.add_scalar("Q_l&time", Q_l, running_time*1000)

    log_data = {
        'Q':Q_l,
        'D':D,
        'accuracy&step':{'x':[],'y':[]},
        'accuracy&time':{'x':[],'y':[]},
        'loss':{'x':[],'y':[]},
        'running_time':{'x':[],'y':[]},
        'communication_volume':{'x':[],'y':[]},
        'CT':{'x':[],'y':[]},
        'commucation_time':0,
        'computation_time':0,
        }

    print(f'server start with batches={len(train_loader)}')

    if True:
        if not predict_shape_list:
            tmp = torch.zeros(2).long()
            predict_shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
            print('gather predict shape..')
            dist.gather(tensor=tmp,gather_list=predict_shape_list)
            print('gather shape ok')

            predict_shape_list.pop(0)
            for i,shape in enumerate(predict_shape_list):
                predict_shape_list[i] = shape.tolist()
            print(predict_shape_list)

            predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,1,predict_shape_list))
            predict_thread.start()

        loss_list = []
        correct_list = []
        acc_list = []
    
        for _, test_target in test_loader:
            predict_h_list = predict_h_list_queue.get()
            for i,h in enumerate(predict_h_list):
                predict_h_list[i] = h.to(device)
            
            predict_y = test_target.to(device)
            loss,correct,accuracy = party.predict(predict_h_list,predict_y)
            loss_list.append(loss)
            correct_list.append(correct)
            acc_list.append(accuracy)
        loss = sum(loss_list) / test_batches
        correct = sum(correct_list) / test_batches
        accuracy = sum(acc_list) / test_batches

        writer.add_scalar("accuracy&step", accuracy, global_step)
        writer.add_scalar("accuracy&time", accuracy, running_time*1000)
        log_data["accuracy&step"]['x'].append(global_step)
        log_data["accuracy&step"]['y'].append(accuracy)
        log_data["accuracy&time"]['x'].append(running_time*1000)
        log_data["accuracy&time"]['y'].append(accuracy)
        print(f'server figure out loss={loss} correct={correct} accuracy={accuracy}\n')

    for ep in range(epochs):
        print(f'server start epoch {ep}')

        for batch_idx, (_, target) in enumerate(train_loader):
            global ts1_wait_time , max_comm_time
            ts1_wait_time = 0
            max_comm_time = 0
            print(f'running_time: {running_time}')
            if global_step >= CT:
                send_data([torch.tensor(-1,dtype=torch.float32) for _ in rank_list],rank_list,tag=1)
                grad_list_queue.put(parties_grad_list)
                is_finish = True
                break
            
            if Cl is None and global_step >= delta_T:
                Cl = (t2 / (Q_l * T_step)) * N_prime * D * Q_l

            if Cl is not None and running_time > last_running_time + Cl:
                t2 = t2 / (Q * T_step)
                gap_time = gap_time / (Q * T_step)
                E = loss_l
                if G is None:
                    G = symbols('G')
                    solution = solve(
                        (-E * ((N * t2)**0.5) / (c0 * ((Cl * N_prime)**0.5) * ((Q_l**3)**0.5))
                         + 2*G * ((t2 * Q_last)**2) * Q_l / (Cl**2)
                         + 9*G * t2 * (D+1) * (Q_l**2) / (Cl * (2*(D**2) + D))
                         ),
                        G
                    )
                    G = solution[0]
                else:
                    Q_last = Q_l
                    f = lambda Q_l:abs(-E * ((N * t2)**0.5) / (c0 * ((Cl * N_prime)**0.5) * ((Q_l**3)**0.5))
                         + 2*G * ((t2 * Q_last)**2) * Q_l / (Cl**2)
                         + 9*G * t2 * (D+1) * (Q_l**2) / (Cl * (2*(D**2) + D))
                         )
                    Q_l = gss(f,0,Q_last*2)
                    # print("Q_l:",Q_l)
                
                # Q = int(Decimal(Q_l).quantize(Decimal("1."), rounding = "ROUND_HALF_UP")) if Q_l > 1 else 1
                # if Q_l > 1:
                #     Q = math.ceil(Q_l) if Q_l - int(Q_l) > np.random.rand() else int(Q_l)
                # else:
                #     Q = 1
                # party.n_iter = Q
                send_data([torch.tensor(Q,dtype=torch.float32) for _ in rank_list],rank_list,tag=1)

                Cl = t2 * N_prime * D * Q_l
                last_t2 = t2 - gap_time
                writer.add_scalar("t2", t2, global_step)
                gap_time = 0
                t2 = 0
                T_step = 0
                last_running_time = running_time
                gap_scale = ((np.random.rand()+0.5)**2-0.25)*10

                writer.add_scalar("Q_l&step", Q_l, global_step)
                writer.add_scalar("Q_l&time", Q_l, running_time*1000)
                writer.add_scalar("Q&time", Q, running_time*1000)
                writer.add_scalar("Cl", Cl, global_step)

            party.model.train()
            start_time = time.time()

            target = target.to(device)
            party.set_batch(target)
            print(f'server set batch {batch_idx}\n')

            if not shape_list:
                tmp = torch.zeros(2).long()
                shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
                print('gather shape..')
                dist.gather(tensor=tmp,gather_list=shape_list)
                print('gather shape ok')

                shape_list.pop(0)
                for i,shape in enumerate(shape_list):
                    shape_list[i] = shape.tolist()
                print(shape_list)

                # pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',h_list_queue,rank_list,0,shape_list))
                # pull_thread.start()

                pull_thread_list = []
                pull_time_thread_list = []
                flag_thread_list =  []
                for rank in rank_list:
                    pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',h_queue_list[rank-1],[rank],0,[shape_list[rank-1]]))
                    pull_thread.start()
                    pull_thread_list.append(pull_thread)
                    if args.use_reweight:
                        pull_time_thread = Thread(target=process_communicate,daemon=True,args=('pull',ts1_queue_list[rank-1],[rank],6,[[]],ts2_queue_list[rank-1]))
                        pull_time_thread.start()
                        pull_time_thread_list.append(pull_time_thread)

                    if args.use_freezing:
                        flag_thread = Thread(target=process_communicate,daemon=True,args=('pull',flag_queue_list[rank-1],[rank],2,[[]]))
                        flag_thread.start()
                        flag_thread_list.append(flag_thread)
            comm_time = 0
            timestamp1 = time.time()
            recv_start_time = time.time()
            # print("h_list_queue",h_list_queue.qsize(),recv_start_time)
            h_list = [torch.zeros(shape) for shape in shape_list]
            # for i in selected_clients:
            #     h_list[i] = h_queue_list[i].get()[0]
            # for h_queue in h_queue_list:
            #     h_list.append(h_queue.get()[0])
            # h_list = h_list_queue.get()
            print(selected_clients)
            for i,q in enumerate(h_queue_list):
                if i in selected_clients:
                    h_list[i] = q.get()[0]
                elif not q.empty():
                    q.get()[0]
                    if h_undo_list[i] > 0:
                        h_undo_list[i] -= 1
            
            # staleness_list = staleness_list_queue.get()
            recv_end_time = time.time()
            recv_spend_time = recv_end_time-recv_start_time
            recv_wait_time += recv_spend_time
            # print('recv spend time: ',recv_spend_time)
            timestamp2 = time.time()

            for i,h in enumerate(h_list):
                h_list[i] = h.to(device)

            party.pull_parties_h(h_list) # concat / avg 得到server的输入h
            party.compute_parties_grad() # 得到返回给每个client的grad
            parties_grad_list = party.send_parties_grad()

            for i,q in enumerate(flag_queue_list):
                if not q.empty():
                    flag = q.get()[0]
                    if flag == 0:
                        freeze_list[i] = False
                        # print(f'flag {i} false')
                    if flag == 1:
                        freeze_list[i] = True
                        # print(f'flag {i} true')
            for i,grad in enumerate(parties_grad_list):
                if freeze_list[i]:
                    parties_grad_list[i] = torch.tensor(337845,dtype=torch.float32)
                else: 
                    parties_grad_list[i] = grad.contiguous().cpu()
            grad_list_queue.put(parties_grad_list)

            # for _ in range(Q):
            #     time.sleep(0.01)
            
            timestamp3 = time.time()
            # gap = np.random.poisson(1) * last_t2 * gap_scale
            # print("gap:",gap)
            # time.sleep(gap)
            timestamp4 = time.time()
            gap_time += timestamp4 - timestamp3

            party.local_update()
            loss = party.get_loss()
            if loss_l is None:
                loss_l = loss
            party.local_iterations()

            end_time = time.time()

            select_start_time = time.time()
            if args.use_reweight and global_step >= args.validation_period + last_validation_step:
                generation_rate_list = torch.zeros(len(rank_list))
                for i,(q1,q2) in enumerate(zip(ts1_queue_list,ts2_queue_list)):
                    if i in selected_clients:
                        ts1 = q1.get()[0]
                        ts2 = q2.get() + get_random_computation_time()[i] + ts1_wait_time
                        wait_time = ts2 - ts1
                        generation_rate_list[i] = wait_time
                        # print(f"h{i} ts1:{ts1}, ts2:{ts2}, wait time: {wait_time}")
                        # generation_rate_list.append(math.exp(-wait_time)) # e^(-t_i)
                        # generation_rate_list.append(wait_time) # t_i
                # generation_rate_list = torch.tensor(generation_rate_list) / sum(generation_rate_list) # t_i list归一化
                print('generation_rate_list:', generation_rate_list)
                
                n = selected_num
                shapley_value_list = torch.zeros(len(rank_list))
                # party.model.train()
                
                def characteristic_function(coalition):
                    if len(coalition) != 0:
                        coalition_h = [h_list[i] for i in coalition]
                    else:
                        coalition_h = [torch.zeros(shape).to(device) for shape in shape_list]
                    h_weight_list = party.h_weight_list
                    party.h_weight_list = [1/len(coalition_h)] * len(coalition_h)
                    party.pull_parties_h(coalition_h)
                    party.compute_parties_grad()
                    loss = party.get_loss().detach().cpu()
                    party.h_weight_list = h_weight_list
                    return loss
                    print(f"value of coalition {coalition} : {math.exp(-loss)}")
                    return math.exp(-loss)
                
                def shapley_value_estimation():
                    party_num = len(selected_clients)
                    # v_list = [0] * (party_num)
                    shapley_value_list = torch.zeros(len(rank_list))
                    all_combinations = list(permutations(selected_clients, party_num))
                    empty_set_loss = characteristic_function([])
                    v_full_set = empty_set_loss - characteristic_function(selected_clients)
                    # print("empty_set_loss:", empty_set_loss)
                    # print("v_full_set:", v_full_set)
                    for _ in range(1, args.sv_sampling_times + 1):
                        v_list = torch.zeros(len(rank_list) + 1)
                        # v_list[0] = v_empty_set
                        coalition = random.choice(all_combinations)
                        # print("coalition:", coalition)
                        # print("threshold:", args.performance_threshold_rate * v_full_set)
                        for i in range(len(coalition)):
                            v_list[i + 1] = empty_set_loss - characteristic_function(coalition[:i+1])
                            if v_list[i + 1] < 0:
                                v_list[i + 1] = 0
                            delta_v = v_list[i + 1] - v_list[i]
                            shapley_value_list[coalition[i]] += delta_v
                            # print(f"i:{i} v_list:{v_list[i + 1]}")
                            # print(f"i:{i} delta_v:{delta_v}")
                            # print(f"i:{i} shapley:{shapley_value_list}")
                            if v_list[i + 1] >= args.performance_threshold_rate * v_full_set:
                                break
                                # print(f"t:{t}, i:{i}, coalition[:i]:{coalition[:i+1]}")
                            
                    # print(compute_times)
                    return shapley_value_list / args.sv_sampling_times

                shapley_value_list = shapley_value_estimation()

                print("shapley值：", shapley_value_list)

                generation_rate_list = generation_rate_list / sum(generation_rate_list)
                for i, t_i in enumerate(generation_rate_list):
                    generation_rate_list[i] = 1 if t_i <= 1 / selected_num else 1 / selected_num / t_i
                shapley_value_list[shapley_value_list < 0] = 0
                # shapley_value_list = shapley_value_list / sum(shapley_value_list)


                print('处理后的generation_rate_list:',generation_rate_list)
                print("处理后的shapley_value_list:", shapley_value_list)
                
                select_times += 1

                # param_a = 0.1
                for i, sv_i, t_i in zip(range(len(shapley_value_list)), shapley_value_list, generation_rate_list):
                    if i in selected_clients:
                        # utility = sv_i * t_i
                        # utility = sv_i
                        avg_contribution[i] = (selected_times_list[i] * avg_contribution[i] + sv_i) / (selected_times_list[i]+1)
                        selected_times_list[i] += 1
                        ucb_values[i] = avg_contribution[i] * t_i
                    exploration_item = args.exploration_weight * math.sqrt(2*math.log(select_times) / selected_times_list[i])
                    ucb_values[i] += exploration_item

                    # if select_times >= warm_up_phase:
                    #     exploration_item = exploration_weight * math.sqrt(2*math.log(select_times) / selected_times_list[i])
                    #     ucb_values[i] = avg_contribution[i] + exploration_item
                        
                
                print("avg_contribution:",avg_contribution)
                print("ucb_values:", ucb_values)

                if select_times >= args.warm_up_phase:
                    # select_num = 5
                    ##print("ucb_values:",ucb_values)
                    # selected_clients = sorted(range(len(ucb_values)), key=lambda i: ucb_values[i], reverse=True)[:select_num]
                    sorted_clients = sorted(range(len(ucb_values)), key=lambda i: ucb_values[i], reverse=True)
                    selected_clients = []
                    contribution_sum = 0
                    for i in sorted_clients:
                        contribution_sum += ucb_values[i]
                        # contribution_sum += avg_contribution[i]
                        selected_clients.append(i)
                        if contribution_sum >= args.selection_threshold_rate * sum(ucb_values):
                        # if contribution_sum >= args.selection_threshold_rate * sum(avg_contribution):
                            break
                    
                    selected_num = len(selected_clients)

                    for i in range(len(party.h_weight_list)):
                        # party.h_weight_list[i] = 1.0 if i in selected_clients else 0.0
                        if i in selected_clients:
                            if unselected_count_list[i] > args.unselected_threshold:
                                party.h_weight_list[i] = args.reweight_beta * math.pow(math.e, -unselected_count_list[i])
                            else:
                                party.h_weight_list[i] = 1.0
                        else:
                            party.h_weight_list[i] = 0
                    party.h_weight_list = party.h_weight_list / sum(party.h_weight_list)
                    
                    tmp = [torch.tensor(1,dtype=torch.float32) if i-1 in selected_clients else torch.tensor(-1,dtype=torch.float32) for i in rank_list]
                    print("selected_clients:",selected_clients)
                    print("selected_num:",selected_num)
                    print("tmp:",tmp)
                    send_data(tmp,rank_list,tag=2)

                    for i, n in enumerate(unselected_count_list):
                        if i in selected_clients:
                            # for _ in range(args.validation_period * n):
                            #     h_queue_list[i].get()
                            unselected_count_list[i] = 0
                        else:
                            unselected_count_list[i] += 1
                    print('unselected_count_list:',unselected_count_list)

                if global_step >= args.warm_up_phase and global_step < 100:
                    args.validation_period = 50
                if global_step >= 100:
                    args.validation_period = (int) (args.validation_alpha * math.pow(global_step, 0.5))
                    # print("args.validation_period:", args.validation_period)

                for i,v in enumerate(h_undo_list):
                    for j in range(v):
                        h_queue_list[i].get()
                    if i in selected_clients:
                        h_undo_list[i] = 0
                    else:
                        h_undo_list[i] = args.validation_period
                last_validation_step = global_step


            # end_time = time.time()
            select_end_time = time.time()
            select_spend_time = select_end_time - select_start_time
            spend_time = end_time - start_time + select_spend_time
            running_time += max([get_random_computation_time()[i] for i in selected_clients]) + max_comm_time
            running_time += spend_time 
            select_time += select_spend_time
            # print(f"spend_time={spend_time} running_time={running_time}")
            t2_total += spend_time - (timestamp2 - timestamp1)
            # t2_first += timestamp1 - start_time + (timestamp3 - timestamp2)
            # t2_last += end_time - timestamp3
            t2 += spend_time - (timestamp2 - timestamp1)
            # print("t2_total",t2_total)

            global_step += 1
            local_step += Q
            T_step += 1

            

            writer.add_scalar("running_time", running_time, global_step)
            writer.add_scalar("communication_volume", comm_volume, global_step)
            writer.add_scalar("select_time", select_time, global_step)
            writer.add_scalar("recv_wait_time", recv_wait_time, global_step)
            writer.add_scalar("loss", loss.detach(), global_step)
            log_data["running_time"]['x'].append(global_step)
            log_data["running_time"]['y'].append(running_time)
            log_data["communication_volume"]['x'].append(global_step)
            log_data["communication_volume"]['y'].append(comm_volume)
            log_data["loss"]['x'].append(global_step)
            log_data["loss"]['y'].append(float(loss.detach()))

            if global_step % recording_period == 0:
                if not predict_shape_list:
                    tmp = torch.zeros(2).long()
                    predict_shape_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
                    print('gather shape..',tmp.shape)
                    dist.gather(tensor=tmp,gather_list=predict_shape_list)
                    print('gather shape ok')

                    predict_shape_list.pop(0)
                    for i,shape in enumerate(predict_shape_list):
                        predict_shape_list[i] = shape.tolist()
                    print(predict_shape_list)

                    predict_thread = Thread(target=process_communicate,daemon=True,args=('pull',predict_h_list_queue,rank_list,1,predict_shape_list))
                    predict_thread.start()

                loss_list = []
                correct_list = []
                acc_list = []

                h_weight_list = party.h_weight_list
                party.h_weight_list = [1/len(predict_shape_list)] * len(predict_shape_list)

                for _, test_target in test_loader:
                    predict_h_list = predict_h_list_queue.get()
                    for i,h in enumerate(predict_h_list):
                        predict_h_list[i] = h.to(device)
                    
                    predict_y = test_target.to(device)
                    loss,correct,accuracy = party.predict(predict_h_list,predict_y)
                    loss_list.append(loss)
                    correct_list.append(correct)
                    acc_list.append(accuracy)
                loss = sum(loss_list) / test_batches
                correct = sum(correct_list) / test_batches
                accuracy = sum(acc_list) / test_batches

                party.h_weight_list = h_weight_list

                writer.add_scalar("accuracy&step", accuracy, global_step)
                writer.add_scalar("accuracy&time", accuracy, running_time*1000)
                log_data["accuracy&step"]['x'].append(global_step)
                log_data["accuracy&step"]['y'].append(accuracy)
                log_data["accuracy&time"]['x'].append(running_time*1000)
                log_data["accuracy&time"]['y'].append(accuracy)
                print(f'server figure out loss={loss} correct={correct} accuracy={accuracy}\n')            

        if is_finish:
            break

    log_data['commucation_time'] = recv_wait_time
    log_data['computation_time'] = t2_total
    
    t2_total = t2_total / local_step
    # print("t2_total",t2_total)
    # t2_first = t2_first / global_step
    # print("t2_first",t2_first)
    # t2_last = t2_last / (local_step - global_step)
    # print("t2_last",t2_last)
        
    timestamp1 = time.time()
    tmp = torch.zeros(1)
    timestamp_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=timestamp_list)
    # print('gather timestamp ok')
    max_timestamp = 0
    for timestamp in timestamp_list:
        max_timestamp = max(max_timestamp,timestamp)
    running_time += max_timestamp - timestamp1
    writer.add_scalar("running_time", running_time, global_step+1)
    # print("running_time",running_time)

    parties_t0_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t0_list)
    # print('gather t0 ok')

    parties_t3_list = [torch.zeros_like(tmp) for _ in range(args.world_size)]
    dist.gather(tensor=tmp,gather_list=parties_t3_list)
    # print('gather t3 ok')

    res_list = [torch.zeros(shape,dtype=torch.double) for shape in shape_list]
    recv_data(res_list,rank_list,tag=2)
    timestamp2 = time.time()
    parties_t1_list = [timestamp2 - timestamp[0][0] for timestamp in res_list]
    # print("parties_t1_list",parties_t1_list)

    max_t0_t1 = 0
    for t0,t1 in zip(parties_t0_list,parties_t1_list):
        max_t0_t1 = max(max_t0_t1,t0 + t1)
    # print("max_t0_t1",max_t0_t1)

    max_t1_Qt3 = 0
    for t1,t3 in zip(parties_t1_list,parties_t3_list):
        max_t1_Qt3 = max(max_t1_Qt3,t1 + party.n_iter * t3)
    # print("max_t1_Qt3",max_t1_Qt3)
    
    for T in range(1,global_step):
        CT = max(max_t0_t1 + T * party.n_iter * t2_total, max_t0_t1 + (T-1) * party.n_iter * t2_total + t2_total + max_t1_Qt3)
        writer.add_scalar("CT", CT, T)
        log_data["CT"]['x'].append(T)
        log_data["CT"]['y'].append(float(CT))
    # print("CT",CT)


    dump_data = json.dumps(log_data)
    with open(os.path.join(log_dir,"log_data.json"), 'w') as file_object:
        file_object.write(dump_data)
    
    writer.close()

def run_client(args):
    rank = args.rank
    ps_rank = 0
    party,train_loader,test_loader,epochs,bound = get_task_data(task_name=args.task_name,id=rank,is_asyn=args.is_asyn,gpu=args.gpu)
    
    set_seed()
    
    print('bound',bound)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    recording_period = 10
    global_step = 0
    last_validation_step = 0
    waiting_grad_num = 0
    t0 = 0
    t1 = 0
    t3 = 0

    shape = None
    predict_shape = None

    is_finish = False
    selected = True

    ###############
    last_theta = None
    wnd_size = train_batches / 2 
    avg_distance = 0
    freeze = False
    freezing_period = wnd_size /2
    freezing_count = 0
    obswnd = deque()
    ###############

    batch_cache = Queue()
    h_queue = Queue()
    grad_queue = Queue()
    predict_h_queue = Queue()
    Ql_queue = Queue()
    select_queue = Queue()
    # staleness_queue = Queue()  # send staleness
    time_queue = Queue()

    send_thread = Thread(target=process_communicate,daemon=True,args=('send',h_queue,[ps_rank],0,None))
    predict_thread = Thread(target=process_communicate,daemon=False,args=('send',predict_h_queue,[ps_rank],1))
    pull_Ql_thread = Thread(target=process_communicate,daemon=True,args=('pull',Ql_queue,[ps_rank],1,[[]]))
    
    # send_staleness_thread = Thread(target=process_communicate,daemon=True,args=('send',staleness_queue,[ps_rank],4))
    send_time_thread = Thread(target=process_communicate,daemon=True,args=('send',time_queue,[ps_rank],6))
    send_thread.start()
    predict_thread.start()
    pull_Ql_thread.start()
    if args.use_reweight:
        pull_select_thread = Thread(target=process_communicate,daemon=True,args=('pull',select_queue,[ps_rank],2,[[]]))
        pull_select_thread.start()
    # send_staleness_thread.start()
    send_time_thread.start()

    # print(f'client start with batches={len(train_loader)}')

    ##################
    log_dir = os.path.join('summary_pic',args.task_name,time.strftime("%Y%m%d-%H%M-p"))
    writer = SummaryWriter(log_dir=log_dir)
    ##################

    if True:
        for test_data, _ in test_loader:
            predict_x = test_data.to(device)
            predict_h = party.predict(predict_x)
            if predict_shape is None:
                predict_shape = torch.tensor(predict_h.shape)
                print('gather predict shape..', predict_shape)
                dist.gather(tensor=predict_shape)
                print('gather shape ok')
            predict_h_queue.put(predict_h.cpu())

    for ep in range(epochs):
        print(f'client start epoch {ep}\n')

        for batch_idx, (data, _) in enumerate(train_loader):
            if not Ql_queue.empty():
                Q_l = int(Ql_queue.get()[0])
                if Q_l == -1:
                    is_finish = True
                    break

                party.n_iter = Q_l
                # print("Q_l:",Q_l)
            
            if not select_queue.empty():
                selected = int(select_queue.get()[0]) == 1
                print("selected:",selected)
                selected = True
            party.model.train()

            timestamp1 = time.time()

            data = data.to(device)
            party.set_batch(data)
            batch_cache.put([batch_idx,data])
            print('batch_idx:',batch_idx)
            # print(f'client set batch {batch_idx}\n')
            
            party.compute_h()
            '''
            ## 模拟计算时间
            random_time = get_random_computation_time()
            # print(random_time)
            print('sleep..', random_time[rank - 1])
            time.sleep(random_time[rank - 1])
            
            timestamp2 = time.time() # timestamp2 - timestamp1 : 计算h的时间
            t0 += timestamp2 - timestamp1
            '''
            timestamp2 = time.time() # timestamp2 - timestamp1 : 计算h的时间
            random_time = get_random_computation_time()
            t0 += timestamp2 - timestamp1 + random_time[rank - 1]
            h = party.get_h()
            timestamp3 = time.time()
            if selected:
                h_queue.put(h.cpu())
            # staleness_queue.put(torch.tensor(waiting_grad_num+1, dtype=torch.float32))

            if shape is None:
                shape = torch.tensor(h.shape)
                print('gather shape..')
                dist.gather(tensor=shape)
                print('gather shape ok')

                pull_thread = Thread(target=process_communicate,daemon=True,args=('pull',grad_queue,[ps_rank],0,[shape.tolist()]))
                pull_thread.start()

            waiting_grad_num += 1

            timestamp4 = time.time()
            t3 += timestamp4 - timestamp3

            while waiting_grad_num > bound or not (grad_queue.empty() or batch_cache.empty()) or (ep == epochs - 1 and batch_idx == train_batches - 1 and waiting_grad_num > 0):
                grad = grad_queue.get()[0].to(device)
                timestamp5 = time.time()
                cache_idx, batch_x = batch_cache.get()

                ###############
                temp_tensor = torch.zeros_like(grad)
                # 将第一个元素赋值为特定值
                temp_tensor[0][0] = 337845
                if not freeze and not torch.equal(temp_tensor, grad):
                    party.set_batch(batch_x)

                    # print(f'client local update with batch {cache_idx}\n')
                    party.compute_h()
                    party.pull_grad(grad)
                    party.local_update()
                    party.local_iterations()
                #############

                timestamp6 = time.time()
                t3 += timestamp6 - timestamp5

                waiting_grad_num -= 1

                ################

                

                if args.use_freezing:
                    if not freeze:
                        current_theta = party.get_theta().detach().cpu()
                        if last_theta is not None:
                            obswnd.append(torch.dist(last_theta,current_theta))
                            if len(obswnd) >= wnd_size:
                                avg_distance = sum(obswnd) / wnd_size
                                print('avg_distance:',avg_distance)
                                obswnd.popleft()
                                if writer is not None:
                                    writer.add_scalar("avg_distance", avg_distance, global_step)

                                # 冻结阈值
                                if avg_distance < args.freezing_threshold:
                                    freeze = True
                                    send_data([torch.tensor(1 if freeze else 0,dtype=torch.float32)],0,tag=2)
                                elif freezing_period > wnd_size/2:
                                        freezing_period -= 1
                                        # print('freezing_period:',freezing_period)
                        last_theta = current_theta.clone()
                    
                    else:
                        freezing_count += 1
                        if freezing_count >= freezing_period:
                            freeze = False
                            freezing_count = 0
                            freezing_period += wnd_size/2
                            send_data([torch.tensor(1 if freeze else 0,dtype=torch.float32)],0,tag=2)
                    
                ###################
                    
                if args.use_reweight and global_step >= args.validation_period + last_validation_step:
                    ts = (time.time()-(timestamp2 - timestamp1)-(timestamp4 - timestamp3)-(timestamp6 - timestamp5))%10000
                    # ts = time.time() % 10000
                    # print("ts1:", ts)
                    time_queue.put(torch.tensor(ts,dtype=torch.float32))
                    # staleness_queue.put(torch.tensor(waiting_grad_num+1, dtype=torch.float32))
                    if global_step >= args.warm_up_phase and global_step < 100:
                        args.validation_period = 50
                    if global_step >= 100:
                        args.validation_period = (int) (args.validation_alpha * math.pow(global_step, 0.5))
                    last_validation_step = global_step

                global_step += 1

                if global_step % recording_period == 0:
                    for test_data, _ in test_loader:
                        predict_x = test_data.to(device)
                        predict_h = party.predict(predict_x)
                        if predict_shape is None:
                            predict_shape = torch.tensor(predict_h.shape)
                            print('gather predict shape..', predict_shape)
                            dist.gather(tensor=predict_shape)
                            print('gather predict shape ok')
                        predict_h_queue.put(predict_h.cpu())
                    if args.use_freezing:
                        send_data([torch.tensor(1 if freeze else 0,dtype=torch.float32)],0,tag=2)    

        if is_finish:
            break
    
    dist.gather(tensor=torch.tensor(time.time()))
    # print('gather timestamp ok')

    t0 = t0 / global_step
    # print("t0",t0)
    t3 = t3 / (global_step * party.n_iter)
    # print("t3",t3)
    dist.gather(tensor=torch.tensor(t0))
    # print('gather t0 ok')
    dist.gather(tensor=torch.tensor(t3))
    # print('gather t3 ok')

    dist.send(tensor=torch.full(shape.tolist(),time.time(),dtype=torch.double),dst=0,tag=2)
    
    predict_h_queue.put(-1)

def process_communicate(task_name,dq,ranks,tag,shape_list=None,output_queue=None):
    print(f'{task_name} thread start\n')

    if type(ranks) is not list:
        ranks = [ranks]
    if type(shape_list) is not list:
        shape_list = [shape_list]
        
    while(True):
        if task_name == 'send':
            data_list = dq.get()
            if type(data_list) is int and data_list==-1:
                break
            if type(data_list) is not list:
                data_list = [data_list]
            send_data(data_list,ranks,tag)
            dq.task_done()
        elif task_name == 'pull':
            res_list = [torch.zeros(shape) for shape in shape_list]
            recv_data(res_list,ranks,tag)
            dq.put(res_list)

        if output_queue:
            output_queue.put(time.time() % 10000)
        
def send_data(data_list,dst_list,tag=0):
    if type(data_list) is not list:
        data_list = [data_list]
    if type(dst_list) is not list:
        dst_list = [dst_list]
    req_list = []
    # print('sending..')
    x = data_list[0]
    size_in_bits = x.element_size() * x.numel() * len(data_list) *8/1024/1024
    lock.acquire()
    global comm_volume
    comm_volume += size_in_bits
    lock.release()
    for i,rank in enumerate(dst_list):
        req_list.append(dist.isend(tensor=data_list[i],dst=rank,tag=tag))            
    for req in req_list:
        req.wait()
    # print('send ok')

def recv_data(res_list,src_list,tag=0):
    if type(res_list) is not list:
        res_list = [res_list]
    if type(src_list) is not list:
        src_list = [src_list]
    req_list = []
    # print('pulling..')
    for i,rank in enumerate(src_list):
        req_list.append(dist.irecv(tensor=res_list[i],src=rank,tag=tag))
    for req in req_list:
        req.wait()
    
    # 通信时间 模拟延迟
    x = res_list[0]
    if tag == 0 and x[0][0] == 337845:
        x = torch.tensor(1.)
    size_in_bits = x.element_size() * x.numel() * len(res_list) *8/1024/1024
    lock.acquire()
    global comm_volume
    comm_volume += size_in_bits
    lock.release()
    comm_time = size_in_bits / bandwidth_mbps
    global ts1_wait_time , max_comm_time
    if tag == 6:
        ts1_wait_time = comm_time
    if comm_time > max_comm_time:
        max_comm_time = comm_time    
    #print("size:", size_in_bits, " time:", comm_time)
    #time.sleep(comm_time)
    #print('pull ok')
    
if __name__ == '__main__':
    main()
