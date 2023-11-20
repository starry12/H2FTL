# -*- coding: UTF-8 -*-
from wiscsim.utils import *
from wiscsim.hostevent import Event, ControlEvent
from workflow import *
import math
from utilities import utils
from ftlsim_commons import Extent
import random
from random import randrange
# from pympler import asizeof

random.seed(1000)

def create_config(ftl_type="dftldes"):
    if ftl_type == "dftldes" or ftl_type == "learnedftl" or ftl_type == "sftl":
        conf = wiscsim.dftldes.Config()
        conf['ftl_type'] = "learnedftl"
        conf['internal_ftl_type'] = ftl_type
    else:
        raise NotImplementedError

    # ssd config
    conf['flash_config']['n_pages_per_block'] = 256
    conf['flash_config']['n_blocks_per_plane'] = 2176
    conf['flash_config']['n_planes_per_chip'] = 1
    conf['flash_config']['n_chips_per_package'] = 1
    conf['flash_config']['n_packages_per_channel'] = 1
    conf['flash_config']['n_channels_per_dev'] = 1

    # set ftl
    conf['do_not_check_gc_setting'] = True
    conf.GC_high_threshold_ratio = 0.96
    conf.GC_low_threshold_ratio = 0.5

    conf['enable_simulation'] = True

    utils.set_exp_metadata(conf, save_data = False,
            expname = 'run_expname',
            subexpname = 'run_subexpname')

    conf['simulator_class'] = 'SimulatorDESNew'

    utils.runtime_update(conf)

    return conf

def split_lpns(offset, size, page_size):
    page_size = float(page_size)
    page_num = 8 * 1024 * 1024
    lpns = [lpn for lpn in range(int(math.floor((offset/page_size) % page_num)), int(math.ceil(((offset+size)/page_size) % page_num)))]

    return lpns

def parse_events(filename, page_size, recorder=True, start_lineno=0, lineno=float('inf'), max_writes = float('inf'), max_write_size = float('inf'), write_only=False, format="MSR", capacity = 2*TB, shift_range=False):
    # capacity = 2*TB --- 这个设置是否有影响? --- 应该没有用到
    if "rocksdb" in filename:
        format = "blktrace"
    if "systor" in filename:
        format = "systor"
    # if "traces" in filename:
    #     format = "normal"
    if "MSR" in filename:
        format = "MSR"
    if "FIU" in filename:
        format = "FIU"
    if "Financial" in filename:
        format = "Financial"
    if "filebench" in filename or "benchbase" in filename:
        format = "blktrace"

    #log_msg("parsing %s with %s format" % (filename, format))
    if recorder:
        events = [ControlEvent(OP_ENABLE_RECORDER)]
    else:
        events = [ControlEvent(OP_DISABLE_RECORDER)]
    # Dict<Format, Tuple<size_scale, time_scale, delimeter>>
    format_config = {"MSR" : (1, 100, ","), "blktrace" : (512, 1000**3, " "), "systor" : (1, 1000**3, ","), "normal" : (1, 1000, " "), "FIU" : (512, 1, " "), "Financial" : (1, 1000**3, ",", 512)} 
    size_scale = format_config[format][0]
    offset_scale = size_scale # MSR_offset_scale = 1
    time_scale = format_config[format][1]
    delimeter = format_config[format][2]
    if len(format_config[format]) > 3:
        offset_scale = format_config[format][3]
    offset_shift = 0
    if shift_range: # false
        offset_shift = random.randint(0, capacity)
        offset_shift = offset_shift // page_size * page_size

    with open(filename) as fp: # open trace
        t_start = None
        last_t = 0
        active_events = 0
        num_writes = 0
        write_size = 0
        exist_lpns = dict()
        #test_workloads = dict()
        test_write = 0
        write_event_num = 0
        test_read = 0
        read_warmup_num = 0
        read_event_num = 0
        warmup_event_num = 0

        #for i in range(int(page_num)):
            #test_workloads[i] = 0
        warm_up_writes = []
        for i, raw in enumerate(fp): # 按行分割，实际上每一行可能涉及到多个page的读/写
            if i < start_lineno:
                continue
            # parse trace
            line = raw.strip().split(delimeter)
            line = list(filter(lambda x: x!= "", line))
            # print(line)
            if format == "MSR":
                t, p, d, mode, offset, size, t0 = line
                t, d, offset, size, t0 = int(t), int(d), int(offset), int(size), int(t0)
            elif format == "normal":
                t, d, offset, size, mode = line
                t, d, offset, size, mode = int(t), int(d), int(offset), int(size), int(mode)
            elif format == "blktrace":
                if not len(line)==11:
                    continue
                a, a2, a3, t, a4, a5, mode, offset, plus, size, a6  = line
                t, offset, size= float(t), int(offset), int(size)
            elif format == "systor":
                if i == 0:
                    continue
                t, t0, mode, d, offset, size = line
                if t0 == "":
                    t0 = 0.0
                t, d, offset, size, t0 = float(t), int(d), int(offset), int(size), float(t0)
            elif format == "Financial":
                app, offset, size, mode, t = line
                if int(app)!=0:
                    continue
                t, offset, size = float(t), int(offset), int(size)
            elif format == "FIU":
                t, pid, proc, offset, size, mode, _, d, _ = line
                t, offset, size = float(t), int(offset), int(size)

            # shift timestamp
            if not t_start:
                t_start = t
            t -= t_start # t = 0

            # scale trace
            offset *= offset_scale # MSR_offset_scale = 1
            size *= size_scale
            t = int(t*time_scale)
            offset += offset_shift # offset_shift = 0
            if size == 0:
                continue

            if mode in ["Read", "R", 0, 'r', "RM"]:
                read_event_num += 1
                if write_only:
                    continue
                op = OP_READ # 读数据op设置为OP_READ
                for lpn in split_lpns(offset, size, page_size):
                    test_read += 1
                '''
                should_warm_up = False
                for lpn in split_lpns(offset, size, page_size):
                    test_read += 1
                    if lpn not in exist_lpns:
                        should_warm_up = True
                        exist_lpns[lpn] = None
                        #test_workloads[lpn] += 1
                if should_warm_up: # 如果当前位置没有数，读取就是错误的，需要先将数据写入，再读取
                    warmup_event_num += 1
                    for lpn in split_lpns(offset, size, page_size):
                        #test_write += 1
                        read_warmup_num += 1
                        #print(lpn)
                    warm_up_writes += [Event(4096, 0, OP_WRITE, offset, size, timestamp=0)] # 创建一个写入缺失数据的lpn的event
                    num_writes += len(split_lpns(offset, size, page_size))
                '''
            elif mode in ["Write", "W", 1, 'w', "WS","WM"]:
                op = OP_WRITE # 写数据op设置为OP_WRITE
                write_event_num += 1
                for lpn in split_lpns(offset, size, page_size):
                    test_write += 1
                    exist_lpns[lpn] = 1
                    #test_workloads[lpn] += 1
                #events += [Event(4096, 0, op, offset, size, timestamp=t)]
                #active_events += 1

            elif mode in ["FN"]:
                continue

            # create event
            if t < last_t:
                continue
            # events += [ControlEvent(OP_SLEEP, arg1=t - last_t)]
            # event用于离散事件模拟
            # offset = lpn * 512


            events += [Event(4096, 0, op, offset, size, timestamp=t)]
            active_events += 1
            last_t = t
        
            # termination
            if i > lineno:
                break

            if num_writes >= max_writes:
                break
            
            if write_size >= max_write_size:
                break

            # if (i-start_lineno) % 1000000 == 0:
            #     log_msg("parsed %d lines" % i)
    
    # timestamp from traces might not be sorted! (now we abort the unsorted ones)
    # events = sorted(events, key=lambda event: event.timestamp)
    # for i in range(0, len(events)):
    #     events.insert(i*2, ControlEvent(OP_SLEEP, arg1=None))
    # last_t = 0
    # for i in range(0, len(events), 2):
    #     sleep, event = events[i], events[i+1] 
    #     sleep.arg1 = event.timestamp - last_t
    #     last_t = event.timestamp

    # wq: Add sequence warm up 
    
    page_num = 8 * 1024 * 1024
    seq_warm_up_ratio = 0
    sequence_warm_up = []
    sequence_warm_up_num = 0
    random_num = 0
    #for i in range(int(page_num * 2)):
        #random_num = random.randint(1, page_num-1)
        #sequence_warm_up += [Event(4096, 0, OP_WRITE, random_num * 4096, 4096, timestamp=0)]
        #sequence_warm_up_num += 1
    
    #for i in range(int(page_num)):
        #sequence_warm_up += [Event(4096, 0, OP_WRITE, i * 4096, 4096, timestamp=0)]
        #sequence_warm_up_num += 1
    
    random_write = []
    random_int_num = []
    random_int_num = random.sample(range(0, page_num), int(8 * 1024 * 1024))
    random_write_num = 0
    for i in random_int_num:
        #random_num = random.randint(1, page_num-1)
        #print(random_num)
        random_write += [Event(4096, 0, OP_WRITE, i * 4096, 4096, timestamp=0)]
        random_write_num += 1
    
    #random_int_num_1 = set(random_int_num)
    #print(len(random_int_num_1))
    

    # ControlEvent(OP_ENABLE_RECORDER)应该是一个用来记录的event
    # 先对缺失的进行
    events = [ControlEvent(OP_ENABLE_RECORDER)] + random_write + events
    #events = [ControlEvent(OP_ENABLE_RECORDER)]
    #print(asizeof.asizeof(events))

    #test_workloads_num = 0
    #for lpn in range(int(page_num)):
        #if test_workloads[lpn] != 0:
            #test_workloads_num += 1

    log_msg("Trace %s" % filename)
    #log_msg("Total warm-up events %d" % len(warm_up_writes)) # 读操作缺失的操作数 --- 实验结果看还是挺多的
    log_msg("Total active events %d" % active_events) # workloads初始自带的操作数（即多少行）
    #log_msg("Total sequence warm up events %d" % sequence_warm_up_num)
    #log_msg("random write %d" % random_write_num)
    
    


    log_msg("test_read_page_num %d" % test_read)
    log_msg("read_event_num %d" % read_event_num)
    #log_msg("read -- write %d" % read_warmup_num)
    #log_msg("warmup_event_num %d" % warmup_event_num)
    log_msg("test_write_num %d" % test_write)
    log_msg("write_event_num %d" % write_event_num)
    log_msg("address space %d" % len(exist_lpns))
    # for lpn in range(int(page_num)):
        # log_msg("lpn %d, num %d" % (lpn, test_workloads[lpn]))
    return events



def split_ext(extent):
    if extent.lpn_count == 0:
        return None

    exts = []
    for lpn in extent.lpn_iter():
        cur_ext = Extent(lpn_start=lpn, lpn_count=1)
        exts.append(cur_ext)

    return exts


def partial_shuffle(l, factor=5):
    n = len(l)
    for _ in range(factor):
        a = randrange(n)
        b = min(n-1, a + randrange(n // 1000))
        l[b], l[a] = l[a], l[b]


def mix_events(all_events, page_size, policy='RR'):
    all_writes = []
    conf = create_config(ftl_type="learnedftl")
    # conf = create_config(ftl_type="learnedftl", page_size=page_size)
    if policy == "RR":
        for i in range(max([len(l) for l in all_events.values()])):
            for t in all_events.keys():
                if len(all_events[t]) > i:
                    event = all_events[t][i]
                    if event.get_operation() != OP_WRITE:
                        continue
                    ext = event.get_lpn_extent(conf)
                    writes = split_ext(ext)
                    all_writes += writes


    if policy in ["RR_Single_Write", "RR_Single_Light_Shuffle"]:
        for trace, events in all_events.items():
            writes = []
            for event in events:
                if event.get_operation() != OP_WRITE:
                    continue
                ext = event.get_lpn_extent(conf)
                writes += split_ext(ext)
            all_events[trace] = writes

        for i in range(sorted([len(l) for l in all_events.values()])[len(all_events)//2]):
            for t in all_events.keys():
                if len(all_events[t]) > i:
                    all_writes.append(all_events[t][i])

        if policy == "RR_Single_Light_Shuffle":
            partial_shuffle(all_writes, len(all_writes) // 200)

    if policy == "RR_Controlled_Random":
        threshold = 0.9
        dist_writes = [len(l) for l in all_events.values()]
        num_of_writes = sum(dist_writes)

        progress = {t : 0 for t in all_events.keys()}
        next_t, i = 0, 0
        while i < num_of_writes:
            tracefile = all_events.keys()[next_t]
            if progress[tracefile] < len(all_events[tracefile]):
                event = all_events[tracefile][progress[tracefile]]
                if event.get_operation() == OP_WRITE:
                    ext = event.get_lpn_extent(conf)
                    writes = split_ext(ext)
                    all_writes += writes
                
                i += 1
                progress[tracefile] += 1

                if random.random() < threshold:
                    next_t += 1
                    next_t %= len(all_events)

            else:
                next_t += 1
                next_t %= len(all_events)    
        

    return all_writes
