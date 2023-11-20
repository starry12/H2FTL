# -*- coding: UTF-8 -*-
from asyncore import write
import bidict
import sys
import copy
from collections import deque, OrderedDict, defaultdict
import datetime
import time
import Queue
import itertools
import math
import objgraph
import pickle
from bisect import bisect_left, insort_left
import numpy as np
import bitarray
import bitarray.util
from wiscsim.utils import *
from wiscsim.sftl import SFTLPage, DFTLPage

import config
import ftlbuilder
from datacache import *
import recorder
from utilities import utils
from .bitmap import FlashBitmap2
from wiscsim.devblockpool import *
from ftlsim_commons import *
from commons import *
import dftldes

import datetime
import numpy
import platform
import collections

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib   # 这三行很重要，否则以命令行的Linux无法生成图像
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LPN_TO_DEBUG = -1

# some constants
LPN_BYTES = 4
SUBLPN_BYTES = 4  # = 1 if use FramePLR
PPN_BYTES = 4
FLOAT16_BYTES = 2
LENGTH_BYTES = 1  # MAX_SEGMENT_LENGTH = 256

CACHE_HIT = 300
COMPACTION_DELAY = 4300
LOOKUP_LATENCY = 50

memory_group = 0


class Ftl(ftlbuilder.FtlBuilder): # dftl，sftl，leaftl的初始化均使用这个类
    # learnedftl.Ftl(self.conf, self.recorder, simpleflash, self.env, self.flash_controller, self.ncq)
    def __init__(self, confobj, recorderobj, flashobj, simpy_env, des_flash, ncq):
        super(Ftl, self).__init__(confobj, recorderobj, flashobj)

        self.des_flash = des_flash # self.store_data = True
        self.env = simpy_env
        #self.ncq = ncq

        self.counter = defaultdict(float)

        self.metadata = FlashMetadata(self.conf, self.counter)
        #self.logical_block_locks = LockPool(self.env)

        self.written_bytes = 0
        #self.discarded_bytes = 0
        self.pre_written_bytes_gc = 0
        self.read_bytes = 0
        self.pre_written_bytes = 0
        #self.pre_discarded_bytes = 0
        self.pre_read_bytes = 0
        self.display_interval = 1000 * MB
        #self.compaction_interval = 100 * MB
        #self.promotion_interval = 500 * MB
        self.gc_interval = 1 * MB
        self.rw_events = 0

        self.rw_cache_hot = RWCache_hot(75*MB, self.conf.page_size, 8*MB, 1.0)
        self.rw_cache_cold = RWCache_cold(75*MB, self.conf.page_size, 8*MB, 1.0)

        self.clean_num = 0
        self.hist = defaultdict(int)
        self.read_latencies = []
        self.write_latencies = []
        self.gc_latencies = []
        self.waf = {"request" : 0, "actual" : 0}
        self.waf_request = 0
        self.waf_actual = 0
        self.raf = {"request" : 0, "actual" : 0}
        self.gc_valid_pages = 0
        self.warmup_num = 0
        self.enable_recording = False
        self.cold_flush_num = 0
        self.hot_flush_num = 0
        self.gc_block_num = 0
        self.gc_tag = 0
        self.cold_hit = 0
        self.hot_hit = 0
        
        self.pb_mapping_table_hit = 0

        self.write_ppn_tag = False


        # @Jie
        self.HAMLconfig = {"MONITORing":False,"Tstarttime":0,"TimePeriod":int(6*60*60*1000*1000),"start_time":int()}
        # 根据论文Time Period设置为6H。
        # 6H对应的时间戳（即微秒数）为6*60*60*1000*1000
        # Tstarttime记录的是原始的18位时间戳，以备以后使用
        # starttime记录的是17位时间戳。

        # TBL的设计：slicenumber:[次数， 平均间隔（初始置为T，最后重新计算），首次访问时间戳， 末次访问时间戳（每次访问都要更新）]
        # 注：python2的字典是无序的，可以输出dict ={"wjk":"Karry","wy":"Roy","yyqx": "Jackson"}试试

        self.TBL = collections.OrderedDict()
        self.HOTNESS = {}
        self.slice_num = int(self.metadata.flash_num_blocks / 100) + 1
        print("slice num % d" % self.slice_num)
        for i in range(int(self.slice_num)):
            self.HOTNESS[i] = 'cold'
        #self.figureindex = 0

    # trace里使用了18位的时间戳，相当于webkit时间戳（17位）+一位的毫秒。
    # 由于本项目每隔数个小时才进行一次聚类，所以暂时不考虑毫秒位。故先舍弃末位
    # def date_from_webkit(webkit_timestamp):
    #     webkit_timestamp = str(webkit_timestamp)[0:-1]
    #     epoch_start = datetime.datetime(1601,1,1)
    #     delta = datetime.timedelta(microseconds=int(webkit_timestamp))
    #     return epoch_start + delta
    
    # @Jie
    def up_to_T_hours(self, webkit_timestamp):
        now_timestamp = int(str(webkit_timestamp)[0:-1])
        interval = now_timestamp-self.HAMLconfig["start_time"]
        return interval>=self.HAMLconfig["TimePeriod"]

    # @Jie
    def run_k_means(self):
        # 先对数据进行预处理
        # key_list = list(self.TBL.keys())
        # for value in self.TBL.values():
        # print(self.TBL)
        for key,value in self.TBL.items():
            if value[0] !=1:
                value[1]=(value[3]-value[2])/10/(value[0]-1)
            else:
                self.HOTNESS[key]='cold'
                del self.TBL[key]   # 这里随即把只访问过一次的slice标记为cold, 并且把这一条目从字典中删除

        # print(self.TBL)

        # @Jie-2：这一块用于三维度的聚类
        extracted_elements = [[sublist[1]] + [sublist[0]] + [1.0*sublist[0]/sublist[4]] for sublist in self.TBL.values()]
        data = np.array(extracted_elements)

        # @Jie-2：所以，之前用来二维度聚类的部分要删掉
        # my_vector = np.array([value[0:2] for value in self.TBL.values()])
        # data = np.array([value[:2][::-1] for value in self.TBL.values()])  #调整了一下顺序，这行代表取列表的前2个元素并且反序，[平均时间间隔, 次数（频率）]
        # print(data)

        # T时间内仅仅访问过一次的Slice提前被定义为Cold，故不需要参与聚类。（如果参与聚类，这个数据在平均时间间隔的这个维度上特别大，不利于聚类）

        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # 进行聚类
        if len(self.TBL)<2:
            return  # 防止极端清空下，比如只剩一个点，进行n_clusters = 2的聚类会报错
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(scaled_data)

        # 获取聚类结果和中心点
        labels = kmeans.labels_
        # centers = kmeans.cluster_centers_

        # 接下来就是问题的关键了...每一批聚类后，都能分成两类，但是哪一类是冷哪一类是热呢？
        # 我们用一个“非常冷”的数据取获得它所在的类，例如用一个间隔为T，次数为1的数据
        # new_data = [[TimePeriod, 1]]
        # predicted_label = kmeans.predict(new_data)
        # 如果predicted_label值为0，则cluster为0的都是冷的。如果另一批结果里面，predicted_label值为1，则在这一批里面，cluster为1的都是冷的
        
        # @Jie-2：
        new_data = [[self.HAMLconfig["TimePeriod"], 1, 300]]
        new_data = scaler.fit_transform(new_data)
        predicted_label = kmeans.predict(new_data)

        # 获得key的索引（因为TBL是有序字典，其中第i个位置的slice的冷热情况是用labels[i]的数来表示的）
        key_list = list(self.TBL.keys())

        if predicted_label == 0:
            for key,value in self.TBL.items():
                if(labels[key_list.index(key)]==0):
                    self.HOTNESS[key]='cold'
                else:
                    self.HOTNESS[key]='hot'
        else:
            for key,value in self.TBL.items():
                if(labels[key_list.index(key)]==0):
                    self.HOTNESS[key]='hot'
                else:
                    self.HOTNESS[key]='cold'
        # 这样写还有一个好处，如果原来的结果在，本次学习的结果不包含原来的结果，相当于是一个补充。
        #print(self.HOTNESS)
        return
    
    def recorder_enabled(self, enable=True):
        self.enable_recording = enable

    def lpn_to_ppn(self, lpn):
        return self.metadata.lpn_to_ppn(lpn)

    def end_ssd(self):
        self.metadata.mapping_table.compact(promote=True)

        cache_lpn_hot, inplace_update_num_hot = self.rw_cache_hot.traversal_unassigned()
        cache_lpn_cold, inplace_update_num_cold = self.rw_cache_cold.traversal_unassigned()
        self.waf['actual'] += cache_lpn_hot
        self.waf['actual'] += cache_lpn_cold
        inplace_update_num = inplace_update_num_cold + inplace_update_num_hot

        log_msg("free block num %d" % len(self.metadata.bvc.free_block_list))
        log_msg("gc block num %d" % self.clean_num)
        log_msg("gc valid pages: %d" % self.gc_valid_pages)
        log_msg("inplace_update_num: %d" % inplace_update_num)

        log_msg("cold_flush_num: %d" % self.cold_flush_num)
        log_msg("hot_flush_num: %d" % self.hot_flush_num)

        log_msg("End-to-end overall response time per page: %.2fus; Num of requests %d" % ((np.sum(self.write_latencies) + np.sum(self.read_latencies)) /  (self.waf["request"] + self.raf['request']), self.waf["request"] + self.raf['request']))

        if len(self.read_latencies) > 0:
            log_msg("End-to-end read response time: %.2fus" % (np.sum(self.read_latencies)))
            log_msg("End-to-end read response time per page: %.2fus; Num of reads %d" % (np.sum(self.read_latencies) / self.raf['request'], self.raf['request']))

        if len(self.write_latencies) > 0:
            log_msg("End-to-end write response time: %.2fus" % (np.sum(self.write_latencies)))
            log_msg("End-to-end write response time per page: %.2fus; Num of writes %d" % (np.sum(self.write_latencies) / self.waf["request"], self.waf['request']))

        log_msg("End-to-end gc response time: %.2fus" % (np.sum(self.gc_latencies)))

        if self.waf['request'] > 0:
            log_msg("Write Amplification Factor: %.2f; Actual: %d; Request: %d" % (self.waf['actual'] / float(self.waf['request']), self.waf['actual'], self.waf['request']))

        self.waf_actual = self.waf['actual'] - 8 * 1024 * 1024
        self.waf_request = self.waf['request'] - 8 * 1024 * 1024
        log_msg("Write Amplification Factor workloads: %.2f; Actual: %d; Request: %d" % (self.waf_actual / float(self.waf_request), self.waf_actual, self.waf_request))

        if self.raf['request'] > 0:
            log_msg("Read Amplification Factor: %.2f; Actual: %d; Request: %d" % (self.raf['actual'] / float(self.raf['request']), self.raf['actual'], self.raf['request']))

        if self.counter['mapping_table_write_miss'] + self.counter['mapping_table_write_hit'] > 0:
            log_msg("Mapping Table Write Miss Ratio: %.2f" % (self.counter['mapping_table_write_miss'] / float(self.counter['mapping_table_write_miss'] + self.counter['mapping_table_write_hit'])))

        if self.counter['mapping_table_read_miss'] + self.counter['mapping_table_read_hit'] > 0:
            log_msg("Mapping Table Read Miss Ratio: %.2f" % (self.counter['mapping_table_read_miss'] / float(self.counter['mapping_table_read_miss'] + self.counter['mapping_table_read_hit'])))

        log_msg(self.counter['pb_mapping_table_read_miss'])
        log_msg(self.counter)
        if sum(self.metadata.levels.values()) > 0:
            log_msg("Avg lookup", sum(self.metadata.levels.values()), sum(int(k)*int(v) for k, v in self.metadata.levels.items()) / float(sum(self.metadata.levels.values())))

        self.recorder.append_to_value_list('distribution of lookups',
                self.metadata.levels)

        
        crb_distribution = []  # crb
        for frame in self.metadata.mapping_table.frames.values():
            crb_size = 0
            for segment in frame.segments:
                if segment.filter:
                    crb_size += len([e for e in segment.filter if e])
            crb_distribution.append(crb_size)

        self.recorder.append_to_value_list('distribution of CRB size',
                crb_distribution)

        if len(crb_distribution) > 0:
            log_msg("CRB avg size: %.2f, CRB 99 size: %.2f, CRB variation: %.2f" % (np.average(crb_distribution), np.percentile(crb_distribution, 99), np.std(crb_distribution)))

    def display_msg(self, mode):
        # log_msg('Event', self.rw_events, 'Read (MB)', self.read_bytes / MB, 'reading', round(float(req_size) / MB, 2))
        display_bytes = 0
        if mode == "Read":
            display_bytes = self.read_bytes
        elif mode == "Write":
            display_bytes = self.written_bytes
        
        #if self.hot_hit != 0:
            #self.hit_miss_ratio_hot = self.hot_miss / self.hot_hit
        #if self.cold_hit != 0:
            #self.hit_miss_ratio_cold = self.cold_miss / self.cold_hit
        print("hot hit", self.hot_hit)
        print("cold hit", self.cold_hit)

        if self.hot_hit > self.cold_hit:  # cache lower
            if self.rw_cache_cold.capacity >= (50 * MB) / (4 * KB):
                print("data cache cold--")
                self.rw_cache_cold.change_capacity(False, 4 * MB)
                self.rw_cache_hot.change_capacity(True, 4 * MB)
        else:
            if self.rw_cache_hot.capacity >= (50 * MB) / (4 * KB):
                print("data cache hot--")
                self.rw_cache_cold.change_capacity(True, 4 * MB)
                self.rw_cache_hot.change_capacity(False, 4 * MB)
        

        # ghost命中越多，需要空间越多
        if self.pb_mapping_table_hit > self.metadata.mapping_table.leaftl_hit & self.metadata.mapping_table.leaftl_size >= 50 * MB: # mapping table lower
            print("leaftl -- ")
            self.metadata.mapping_table.leaftl_size -= 4 * MB
            self.metadata.pb_mapping_table_size += 4 * MB
        #else:
            #if self.metadata.pb_mapping_table_size >= 4 * MB:
                #print("pb mapping table -- ")
                #self.metadata.mapping_table.leaftl_size += 4 * MB
                #self.metadata.pb_mapping_table_size -= 4 * MB
        

        if (self.hot_hit + self.cold_hit) * 2 > self.pb_mapping_table_hit + self.metadata.mapping_table.leaftl_hit: # upper
            if self.metadata.mapping_table.leaftl_size >= 50 * MB:
                print("hot cache size ++, leaftl --")
                self.rw_cache_hot.change_capacity(True, 4 * MB)
                self.metadata.mapping_table.leaftl_size -= 4 * MB
            else:
                if self.metadata.pb_mapping_table_size >= 8 * MB:
                    print("hot cache size ++, pb mapping table --")
                    self.rw_cache_hot.change_capacity(True, 4 * MB)
                    self.metadata.pb_mapping_table_size -= 4 * MB
        else:
            if self.rw_cache_cold.capacity >= (50 * MB) / (4 * KB):
                print("pb mapping table ++, data cache cold-- ")
                self.metadata.pb_mapping_table_size += 4 * MB
                self.rw_cache_cold.change_capacity(False, 4 * MB)
            else:
                if self.rw_cache_hot.capacity >= (50 * MB) / (4 * KB):
                    print("pb mapping table ++, data cache hot-- ")
                    self.metadata.pb_mapping_table_size += 4 * MB
                    self.rw_cache_hot.change_capacity(False, 4 * MB)


        print("pb_mapping_table_size", self.metadata.pb_mapping_table_size)
        print("leaftl size", self.metadata.mapping_table.leaftl_size)
        print("hot data cache size", self.rw_cache_hot.capacity * 4 * 1024)
        print("cold data cache size", self.rw_cache_cold.capacity * 4 * 1024)
        print(self.waf["request"])


        self.hot_hit = 0
        self.cold_hit = 0
        self.pb_mapping_table_hit = 0
        self.metadata.mapping_table.leaftl_hit = 0


        avg_lookup = 0
        if float(sum(self.metadata.levels.values())) != 0:
            avg_lookup = sum(int(k)*int(v) for k, v in self.metadata.levels.items()) / float(sum(self.metadata.levels.values()))
        log_msg('Event', self.rw_events, '%s (MB)' % mode, display_bytes / MB, "Mapping Table", self.metadata.mapping_table.memory, "Reference Mapping Table", self.metadata.reference_mapping_table.memory, "Distribution of lookups", self.metadata.levels[1], sum(self.metadata.levels.values()), avg_lookup, "Misprediction", self.hist, "Latency per page: %.2fus" % ((np.sum(self.write_latencies) + np.sum(self.read_latencies)) /  (self.waf["request"] + self.raf['request'])))
        sys.stdout.flush()

    def read_ext(self, extent): #读操作
        should_print = False
        req_size = extent.lpn_count * self.conf.page_size
        self.recorder.add_to_general_accumulater('traffic', 'read', req_size)
        self.read_bytes += req_size
        self.rw_events += 1
        if self.read_bytes > self.pre_read_bytes + self.display_interval: #每1000MB输出一次信息
            self.display_msg("Read")
            self.pre_read_bytes = self.read_bytes

        extents = split_ext(extent) # 将workloads一行操作分离成page为粒度的操作
        start_time = self.env.now # <----- start
            
        op_id = self.recorder.get_unique_num()
        # if op_id == 279353:
        #     should_print = True
        procs = []
        total_pages_to_read = []  # 为什么会有read和write两个队列，正常一行的操作不应该只有一个吗?
        total_pages_to_write = [] # --- 因为可能出现data cache满了需要替换的情况，在这种情况下，需要把替换出的page写入flash  # 倾向于未使用
        requested_read = 0.0

        if should_print:
            log_msg(start_time)

        for ext in extents: # 遍历所有lpn
             # nothing to lookup if no lpn is written
            if not self.metadata.reference_mapping_table.get(ext.lpn_start): #不在reference_mapping table里
                if ext.lpn_start not in self.metadata.page_based_mapping_table:
                    self.counter['RAW'] += 1
                    continue

            assert(ext.lpn_count == 1)
            requested_read += 1

            cachehit, writeback, ppn = self.rw_cache_cold.read(ext.lpn_start) # 看是否在data cache里

            if not cachehit: # 不在cold cache
                evict_cold_hit = self.rw_cache_cold.read_evict_cold(ext.lpn_start)
                if evict_cold_hit:
                    self.cold_hit += 1
                overall_cache_capacity = self.rw_cache_cold.capacity + self.rw_cache_hot.capacity
                cachehit, writeback, ppn = self.rw_cache_hot.read(ext.lpn_start, overall_cache_capacity)
            #else:
                
                
            if not cachehit: #不在data cache里
                evict_hot_hit = self.rw_cache_hot.read_evict_hot(ext.lpn_start)
                if evict_hot_hit:
                    self.hot_hit += 1

                if ext.lpn_start in self.metadata.page_based_mapping_table:
                    if ext.lpn_start in self.metadata.pb_mapping_table_cache:
                        self.counter["pb_mapping_table_read_hit"] += 1
                        ppn_start = self.metadata.pb_mapping_table_cache[ext.lpn_start]
                        self.metadata.pb_mapping_table_cache.move_to_head(ext.lpn_start, ppn_start)
                        yield self.env.process(self._read_ppns([ppn_start]))
                    else:
                        self.pb_mapping_table_hit += 1 # 命中在ssd的部分即认为命中在ghost部分
                        self.counter["pb_mapping_table_read_miss"] += 1
                        ppn_start = self.metadata.pb_mapping_table_storage[ext.lpn_start]
                        self.metadata.pb_mapping_table_cache[ext.lpn_start] = ppn_start
                        del self.metadata.pb_mapping_table_storage[ext.lpn_start]
                        yield self.env.process(self._read_ppns([ppn_start]))
                        yield self.env.process(self._read_ppns([0]))

                        if len(self.metadata.pb_mapping_table_cache) * 8 > self.metadata.pb_mapping_table_size:
                            victim_key, victim_value = self.metadata.pb_mapping_table_cache.popitem(last=False)
                            self.metadata.pb_mapping_table_storage[victim_key] = victim_value
                            #del self.metadata.pb_mapping_table_cache[victim_key]
                else:
                    procs += [self.env.process(self.read_logical_block(ext, should_print))]
            else:
                self.hist[0] += 1
                yield self.env.timeout(CACHE_HIT)

            if writeback:
                self.waf["actual"] += 1
                if self.write_ppn_tag == True:
                    log_msg("write ppn 1")
                yield self.env.process(self._write_ppns([ppn]))
                total_pages_to_write.append(ppn)

        # self.env.exit(ext_data)
        if should_print:
            log_msg(requested_read)
            log_msg(self.env.active_process)

        ret = yield simpy.AllOf(self.env, procs)

        total_pages_to_read += [page for pages in ret.values() for page in pages[1]]
        total_pages_to_write += [page for pages in ret.values() for page in pages[0]]

        lpns_to_read = float(len(total_pages_to_read))
    
        end_time = self.env.now

        if self.enable_recording:
            if requested_read > 0:
                self.read_latencies += [(end_time - start_time) / 1000.0] # [(end_time - start_time)/(1000.0*requested_read)]*int(requested_read)

                self.raf["request"] += requested_read
                self.raf["actual"] += lpns_to_read
                # self.waf["actual"] += len(total_pages_to_write) # change waf['actual']

                write_timeline(self.conf, self.recorder,
                    op_id = op_id, op = 'read_ext', arg = extent.lpn_count,
                    start_time = start_time, end_time = end_time)

    def read_logical_block(self, extent, should_print=False): # 输入的extent代表一个原始lpn
        assert(extent.lpn_count == 1)

        # replace the following lines with a nice interface
        lpn = extent.lpn_start
        read_ppns = []

        ppn, pages_to_write, pages_to_read = self.metadata.lpn_to_ppn(lpn)

        if ppn:
            # if accurate mapping entry
            if ppn not in pages_to_read:
                pages_to_read.append(ppn)

            self.hist[len(pages_to_read)] += 1
            block_id = ppn / self.conf.n_pages_per_block

            procs = []
            for read_ppn in pages_to_read:
                if should_print:
                    log_msg(read_ppn, self.env.now)
                yield self.env.process(self._read_ppns([read_ppn]))

            for write_ppn in pages_to_write:
                self.waf["actual"] += 1
                if should_print:
                    log_msg(write_ppn, self.env.now)
                if self.write_ppn_tag == True:
                    log_msg("write ppn 2")
                yield self.env.process(self._write_ppns([write_ppn]))

            # yield simpy.AllOf(self.env, procs)
            if should_print:
                log_msg("Hi", pages_to_read, self.env.now)

        self.env.exit((pages_to_write, pages_to_read))

    def lba_write(self, lpn, data=None):
        yield self.env.process(
            self.write_ext(Extent(lpn_start=lpn, lpn_count=1), [data]))

    def write_ext(self, extent, data=None): #写操作
        req_size = extent.lpn_count * self.conf.page_size
        self.recorder.add_to_general_accumulater('traffic', 'write', req_size)
        self.written_bytes += req_size
        self.rw_events += 1

        self.warmup_num += 1

        if self.warmup_num == 8 * 1024 * 1024:
            #self.write_ppn_tag = True
            log_msg("free block num %d" % len(self.metadata.bvc.free_block_list))
            log_msg("gc block num %d" % self.clean_num)
            log_msg("gc valid pages: %d" % self.gc_valid_pages)
            log_msg("ppn_with_lpn: %d" % len(self.metadata.ppn_with_lpn))
            log_msg(self.counter)
            log_msg("End-to-end read response time: %.2fus" % (np.sum(self.read_latencies)))
            log_msg("End-to-end write response time: %.2fus" % (np.sum(self.write_latencies)))
            log_msg("End-to-end gc response time: %.2fus" % (np.sum(self.gc_latencies)))
            log_msg("Write Amplification Factor: %.2f; Actual: %d; Request: %d" % (self.waf['actual'] / float(self.waf['request']), self.waf['actual'], self.waf['request']))

        if self.written_bytes > self.pre_written_bytes + self.display_interval: # 论文内容：每each 1 million writes进行compaction --- 实现：每1000MB进行compaction 
            self.metadata.mapping_table.compact() # mapping_table的类型为FrameLogPLR()
            self.metadata.mapping_table.promote()

            self.display_msg("Write")
            self.pre_written_bytes = self.written_bytes
            log_msg("free block num %d" % len(self.metadata.bvc.free_block_list))
            
            yield self.env.timeout(COMPACTION_DELAY) # COMPACTION_DELAY = 4300 --- 意思是compaction的时间假设是固定的?

        extents = split_ext(extent) #将写入分成lpn列表

        start_time = self.env.now # <----- start
        write_procs = []
        total_pages_to_write = []

        # @Jie
        # 每一个请求都会被细分成若干extents，因此，extents的大小（元素个数）实际上就是这个请求访问的次数
        # 预设每个切片的大小是4K个LPN

        # 在没有开始Monitor，但是出现不属于warmup的写请求的时候，则开始moniter并记录Tstarttime
        
        if self.HAMLconfig["MONITORing"]==False and extent.timestamp != 0:
            self.HAMLconfig["MONITORing"]=True
            self.HAMLconfig["Tstarttime"]=extent.timestamp
            self.HAMLconfig["start_time"]=int(str(self.HAMLconfig["Tstarttime"])[0:-1])

        # 如果已经开始Monitor，则计算当前时间和开始时间的时间差。如果时间差达到6小时，则进行一次聚类
        elif self.HAMLconfig["MONITORing"]==True:
            if self.up_to_T_hours(int(extent.timestamp)):
                #print('------------Start-----------RUNNing-------------KMEANS--------------')
                self.run_k_means()
                #print('------------<END>--------------')
                # 本轮聚类完毕，初始化（清空）UpdateF_TBL和UpdateT_TBL
                self.HAMLconfig["MONITORing"]=False
                self.TBL = {}


        # 关于平均时间间隔的计算方法（与原文有较大差异）
        # 本文记录每个slice在指定T内的首次时间和末次时间（末次时间每当有新的访问就会被覆盖）
        # 最后，时间到达T的时候，对于某个slice，如果访问次数为1，则把它的平均间隔数值置为T，否则，用末次时间减去首次时间再除以次数
        

        # @Jie-2：改成了枚举
        for i, ext in enumerate(extents): #遍历每个写入的lpn
            # @Jie
            # 根据MONITORing字段的值，只有True的时候才修改UpdateF TBL和UpdateT TBL
            if ext.timestamp != 0:
                slice_no=int(ext.lpn_start)/(25*KB) # 每个slice100M，每个page4K，因此每个slice包含25K个page。
                # print(slice_no)
                if slice_no not in self.TBL:
                    self.TBL.update({slice_no:[1, self.HAMLconfig["TimePeriod"], int(ext.timestamp), int(ext.timestamp)]})
                    # print(self.TBL)
                    
                else:
                    UpdateF = int(self.TBL[slice_no][0])+1
                    self.TBL[slice_no][0] = UpdateF     #更新访问次数（+1）
                    self.TBL[slice_no][3] = ext.timestamp   #更新末次访问时间

                # @Jie-2：
                if i==0:
                    if len(self.TBL[slice_no])==4:
                        self.TBL[slice_no].append(1)#追加一个字段，用来存储同一个slice里面的extent的数量（不是页面数量！！！）
                    else:
                        self.TBL[slice_no][4]+=1

            # @Jie:对所有的字典的访问之前都应该先判断该键是否存在，否则可能有潜在的问题
            if int(ext.lpn_start)/(25*KB) not in self.HOTNESS:
                overall_cache_capacity = self.rw_cache_cold.capacity + self.rw_cache_hot.capacity
                hit, writeback, ppn = self.rw_cache_cold.write(ext.lpn_start, overall_cache_capacity, hotness = 'cold')
                if hit == False:
                    self.rw_cache_hot.delete(ext.lpn_start)
            else:
                if self.HOTNESS[int(ext.lpn_start)/(25*KB)] == 'cold':
                    overall_cache_capacity = self.rw_cache_cold.capacity + self.rw_cache_hot.capacity
                    hit, writeback, ppn = self.rw_cache_cold.write(ext.lpn_start, overall_cache_capacity, hotness = 'cold') # 写入data cache / data buffer
                    if hit == False:
                        self.rw_cache_hot.delete(ext.lpn_start)
                        write_cold_hit = self.rw_cache_cold.read_evict_cold(ext.lpn_start)
                        if write_cold_hit:
                            self.cold_hit += 1
                else:
                    overall_cache_capacity = self.rw_cache_cold.capacity + self.rw_cache_hot.capacity
                    hit, writeback, ppn = self.rw_cache_hot.write(ext.lpn_start, overall_cache_capacity)
                    if hit == False:
                        self.rw_cache_cold.delete(ext.lpn_start)
                        write_hot_hit = self.rw_cache_hot.read_evict_hot(ext.lpn_start)
                        if write_hot_hit:
                            self.hot_hit += 1

            if writeback: # 需要从cache踢数据并写回
                self.waf["actual"] += 1
                if self.write_ppn_tag == True:
                    log_msg("write ppn 3")
                p = self.env.process(self._write_ppns([ppn])) # _write_ppns才是真实写入的事件
                write_procs.append(p)
                total_pages_to_write.append(ppn)

            if self.rw_cache_cold.should_assign_page_cold() or self.rw_cache_hot.should_assign_page_hot(): # cache和unassigned都满了
                self.gc_block_num += 1
                if self.rw_cache_cold.should_assign_page_cold(): # cold --leaftl
                    exts = self.rw_cache_cold.flush_unassigned(hotness = 'cold') #返回分组后的需要flush的exts
                    self.cold_flush_num += 1
                    mappings, pages_to_read, pages_to_write = self.metadata.update(exts) # 分配ppn, 更新pvb, 更新mapping table
                    self.rw_cache_cold.update_assigned(mappings) # flush的同时更改dirty_page_mapping
                else:
                    exts = self.rw_cache_hot.flush_unassigned(hotness = 'hot')
                    self.hot_flush_num += 1
                    mappings, pages_to_read, pages_to_write = self.metadata.update_pb(exts)
                    self.rw_cache_hot.update_assigned(mappings)

                for ppn in pages_to_write: # 这里并不是真正flush到flash里,而是构建一个dirty_mapping_table,只有被踢出cache时才是真正写入cache的
                    self.waf["actual"] += 1
                    if self.write_ppn_tag == True:
                        log_msg("write ppn 5")
                    p = self.env.process(self._write_ppns([ppn]))
                    write_procs.append(p)

                for ppn in pages_to_read:
                    p = self.env.process(self._read_ppns([ppn]))
                    write_procs.append(p)
            
            if not writeback:
                yield self.env.timeout(CACHE_HIT)

        yield simpy.AllOf(self.env, write_procs) # events中所有event被触发

        end_time = self.env.now # <----- end

        if self.enable_recording:
            self.write_latencies.append((end_time - start_time)/1000.0) #us
            self.waf["request"] += extent.lpn_count # lpn_count == 1    

    def _write_ppns(self, ppns):
        """
        The ppns in mappings is obtained from loggroup.next_ppns()
        """
        # flash controller
        yield self.env.process(
            self.des_flash.rw_ppns(ppns, 'write',
                                   tag="Unknown"))

        self.env.exit((0, 0))

    def _read_ppns(self, ppns):
        """
        The ppns in mappings is obtained from loggroup.next_ppns()
        """
        # flash controller
        yield self.env.process(
            self.des_flash.rw_ppns(ppns, 'read',
                                   tag="Unknown"))

    def clean(self, forced=False, merge=True): # gc --- 有效页面数在90%以下时擦除所有有效页面小于10%的block
        self.pre_written_bytes_gc = self.written_bytes
        erased_pbns = []
        validate_pages = []
  
        block_num = 256 + 424

        if len(self.metadata.bvc.free_block_list) > block_num:
            return
        
        if(self.gc_tag == 0):
            self.gc_block_num = 1
            self.gc_tag = 1
        
        num_valid_pages = 256
        pick_block = 0
        valid_ppn = []
        valid_lpn = []
        start = self.env.now

        for i in range(int(8)):
            for block in self.metadata.bvc.counter:
                if self.metadata.bvc.counter[block] <= num_valid_pages and self.metadata.bvc.counter[block] != 0:
                    num_valid_pages = self.metadata.bvc.counter[block]
                    pick_block = block
            erased_pbns.append(pick_block)
            self.metadata.add_block(pick_block)
            valid_ppn = self.metadata.pvb.get_valid_pages(pick_block)
            for ppn in valid_ppn:
                self.metadata.pvb.bitmap[ppn] = self.metadata.pvb.INVALID
                if ppn not in self.metadata.ppn_with_lpn:
                    print("not exist")
                target_lpn = self.metadata.ppn_with_lpn[ppn]
                valid_lpn.append(target_lpn)
                self.rw_cache_hot.read_and_delete(target_lpn)
                self.rw_cache_cold.read_and_delete(target_lpn)
            all_pages = self.metadata.pvb.get_pages(pick_block)
            for ppn in all_pages:
                if ppn not in self.metadata.ppn_with_lpn:
                    print("not exist", ppn)
                else:
                    del self.metadata.ppn_with_lpn[ppn]
            self.clean_num += 1
            validate_pages += valid_lpn
            self.gc_valid_pages += num_valid_pages
            num_valid_pages = 256
            valid_lpn = []
        self.gc_block_num = 0

        write_procs = []
        for lpn in validate_pages: #遍历有效页面
            if self.HOTNESS[int(lpn)/(25*KB)] == 'cold':
                overall_cache_capacity = self.rw_cache_cold.capacity + self.rw_cache_hot.capacity
                hit, writeback, ppn = self.rw_cache_cold.write(lpn, overall_cache_capacity, hotness = 'cold') 
                if hit == False:
                    self.rw_cache_hot.delete(lpn)
                    write_cold_hit = self.rw_cache_cold.read_evict_cold(lpn)
                    if write_cold_hit:
                        self.cold_hit += 1
            else:
                overall_cache_capacity = self.rw_cache_hot.capacity + self.rw_cache_cold.capacity
                hit, writeback, ppn = self.rw_cache_hot.write(lpn, overall_cache_capacity)
                if hit == False:
                    self.rw_cache_cold.delete(lpn)
                    write_hot_hit = self.rw_cache_hot.read_evict_hot(lpn)
                    if write_hot_hit:
                        self.hot_hit += 1


            if writeback: # 需要从cache踢数据并写回
                self.waf["actual"] += 1
                if self.write_ppn_tag == True:
                    log_msg("write ppn 7")
                p = self.env.process(self._write_ppns([ppn])) # _write_ppns才是真实写入的事件
                write_procs.append(p)

            if self.rw_cache_cold.should_assign_page_cold() or self.rw_cache_hot.should_assign_page_hot(): # cache和unassigned都满了
                self.gc_block_num += 1
                if self.rw_cache_cold.should_assign_page_cold(): # cold --leaftl
                    exts = self.rw_cache_cold.flush_unassigned(hotness = 'cold') #返回分组后的需要flush的exts
                    self.cold_flush_num += 1
                    mappings, pages_to_read, pages_to_write = self.metadata.update(exts) # 分配ppn, 更新pvb, 更新mapping table
                    self.rw_cache_cold.update_assigned(mappings) # flush的同时更改dirty_page_mapping
                else:
                    exts = self.rw_cache_hot.flush_unassigned(hotness = 'hot')
                    self.hot_flush_num += 1
                    mappings, pages_to_read, pages_to_write = self.metadata.update_pb(exts)
                    self.rw_cache_hot.update_assigned(mappings)
                
                for ppn in pages_to_write: # 这里并不是真正flush到flash里,而是构建一个dirty_mapping_table,只有被踢出cache时才是真正写入cache的
                    self.waf["actual"] += 1
                    if self.write_ppn_tag == True:
                        log_msg("write ppn 9")
                    p = self.env.process(self._write_ppns([ppn]))
                    write_procs.append(p)

                for ppn in pages_to_read:
                    p = self.env.process(self._read_ppns([ppn]))
                    write_procs.append(p)

            if not writeback:
                yield self.env.timeout(CACHE_HIT)
        
        erase_procs = []
        for erased_pbn in erased_pbns: #遍历所有需要擦除的block
            erase_procs += [self.env.process(self.des_flash.erase_pbn_extent(pbn_start = erased_pbn, pbn_count = 1, tag = None))] # 进行machine擦除操作
            
        
        yield simpy.AllOf(self.env, erase_procs)
        erase_finished = self.env.now
        yield simpy.AllOf(self.env, write_procs)
        write_finished = self.env.now
        self.gc_latencies.append((write_finished - start)/1000.0)
        # print(len(validate_pages), erase_finished - start, write_finished - erase_finished)

    def is_cleaning_needed(self):
        if self.written_bytes > self.pre_written_bytes_gc + self.gc_interval:
            return True
        return False

    def snapshot_erasure_count_dist(self):
        dist = self.block_pool.get_erasure_count_dist()
        self.recorder.append_to_value_list('ftl_func_erasure_count_dist',
                                           dist)
    
    def snapshot_user_traffic(self):
        return


class PageValidityBitmap(object):
    "Using one bit to represent state of a page"
    "Erased state is recorded by BVC"
    VALID, INVALID = (1, 0)

    def __init__(self, conf, bvc):
        if not isinstance(conf, config.Config):
            raise TypeError("conf is not conf.Config. it is {}".
                            format(type(conf).__name__))

        self.conf = conf
        self.bitmap = bitarray.bitarray(conf.total_num_pages())
        self.bitmap.setall(0)
        self.bvc = bvc

    def validate_page(self, pagenum):
        self.bitmap[pagenum] = self.VALID
        self.bvc.counter[pagenum // self.conf.n_pages_per_block] += 1

    def invalidate_page(self, pagenum):
        if self.bvc.counter[pagenum // self.conf.n_pages_per_block] != 0:
            self.bitmap[pagenum] = self.INVALID
            self.bvc.counter[pagenum // self.conf.n_pages_per_block] -= 1
        

    def validate_block(self, blocknum):
        ppn_start, ppn_end = self.conf.block_to_page_range(blocknum)
        for pg in range(ppn_start, ppn_end):
            self.validate_page(pg)

    def get_num_valid_pages(self, blocknum):
        start, end = self.conf.block_to_page_range(blocknum)
        return sum(self.bitmap[start:end])

    def get_valid_pages(self, blocknum):
        valid_pages = []
        start, end = self.conf.block_to_page_range(blocknum)
        for pg in range(start, end):
            if self.bitmap[pg]:
                valid_pages.append(pg)
        return valid_pages
    
    def get_pages(self, blocknum):
        all_pages = []
        start, end = self.conf.block_to_page_range(blocknum)
        for pg in range(start, end):
            all_pages.append(pg)
        return all_pages


    def is_page_valid(self, pagenum):
        return self.bitmap[pagenum] == self.VALID

    def is_page_invalid(self, pagenum):
        return self.bitmap[pagenum] == self.INVALID

    @property
    def memory(self):
        return round(len(self.bitmap) // 8)


class BlockValidityCounter(object):
    """
        Timestamp table PPN -> timestamp
        Here are the rules:
        1. only programming a PPN updates the timestamp of PPN
           if the content is new from FS, timestamp is the timestamp of the
           LPN
           if the content is copied from other flash block, timestamp is the
           same as the previous ppn
        2. discarding, and reading a ppn does not change it.
        3. erasing a block will remove all the timestamps of the block
        4. so cur_timestamp can only be advanced by LBA operations
        Table PPN -> valid pages
    """

    def __init__(self, conf):
        self.conf = conf
        #self.last_inv_time_of_block = {}
        self.timestamp_table = {}
        self.cur_timestamp = 0
        self.counter = defaultdict(lambda:0)
        self.free_block_list = [self.conf.n_blocks_per_channel * channel + block 
                                for block in range(self.conf.n_blocks_per_channel)
                                for channel in range(self.conf.n_channels_per_dev)]
        print("free block num %d" % len(self.free_block_list))


    def _incr_timestamp(self):
        """
        This function will advance timestamp
        """
        t = self.cur_timestamp
        self.cur_timestamp += 1
        return t

    def set_timestamp_of_ppn(self, ppn):
        self.timestamp_table[ppn] = self._incr_timestamp()

    def copy_timestamp(self, src_ppn, dst_ppn):
        self.timestamp_table[dst_ppn] = self.timestamp_table[src_ppn]

    def get_num_valid_pages(self, blocknum):
        return self.counter[blocknum]

    # FIXME improve efficiency; now round-robin
    # bottleneck
    def next_free_block(self, wear_level=False):
        free_block = self.free_block_list.pop(0)
        return free_block

    def gc_block(self, blocknum):
        self.counter[blocknum] = 0
        self.free_block_list.append(blocknum)

class OutOfBandAreasMemOpt(object):
    """
    Memory optimized version
    Jinghan: We use OOB to store the p2l mapping for each page and all PPNs within the same segment. Since we do not update the segments in-place, we also do not have to update the OOB data.
    OOB impl: Dict<ppn, Dict<ppn, lpn>>
    Real OOB: List<lpn>
    """

    def __init__(self, conf, gamma, reference_mapping_table):
        self.gamma = gamma
        self.num_p2l_entries = min(2*self.gamma, conf['flash_config']["oob_size_per_page"] / 4)
        self.reference_mapping_table = reference_mapping_table

    # entries: List<Tuple<lpn, ppn>>
    def set_oob(self, source_page, entries):
        pass

    def ppn_to_lpn(self, ppn, source_page=None):
        pass

    def lpn_to_ppn(self, lpn, source_page):
        real_ppn = self.reference_mapping_table.get(lpn)
        if abs(real_ppn - source_page) <= self.num_p2l_entries:
            return real_ppn
        return None

# self.metadata
class FlashMetadata(object):
    def __init__(self, confobj, counter): # 疑问： 没有Global Mapping Directory (GMD)? --- 如果不在memory里cache，如何知道对应的存储mapping table的物理位置?  --- gtd
        self.conf = confobj
        self.counter = counter

        self.flash_num_blocks = confobj.n_blocks_per_dev
        self.flash_npage_per_block = confobj.n_pages_per_block
        self.total_pages = self.flash_num_blocks * self.flash_npage_per_block

        self.gamma = self.conf['gamma']
        self.mapping_table = FrameLogPLR(confobj, self, counter, gamma=self.gamma)
        self.pb_mapping_table_size = 114 * 1024 * 1024

        self.page_based_mapping_table = dict() # page_based mapping table
        self.pb_mapping_table_cache = LRUCache()
        self.pb_mapping_table_storage = dict()

        Segment.PAGE_PER_BLOCK = self.flash_npage_per_block
        self.reference_mapping_table = PFTL() # 是指以前那种1-1的mapping table?
        self.bvc = BlockValidityCounter(confobj)
        self.pvb = PageValidityBitmap(confobj, self.bvc)
        self.oob = OutOfBandAreasMemOpt(confobj, self.gamma, self.reference_mapping_table)
        self.last_oob_page = []
        self.next_free_ppn = 0
        self.waf = []
        self.now_pb_mapping_table_cache_size = 0

        self.levels = defaultdict(int)
        self.ppn_with_lpn = dict()


    def add_block(self, blocknum):
        self.bvc.gc_block(blocknum)

    ############# Flash read related ############

    def ppn_to_lpn(self, ppn, source_page=None):
        return self.oob.ppn_to_lpn(ppn, source_page=source_page)

    ############# Flash write related ############

    def lpn_to_ppn(self, lpn): # 输入lpn为初始lpn
        real_ppn = None
        results, num_lookup, pages_to_write, pages_to_read = self.mapping_table.lookup(lpn, first=True) # 这里是去找mapping_table，也就是成组的log-structured结构
        self.levels[num_lookup] += 1

        if len(results) > 0: # 找到ppn之后
            ppn, accurate, seg = results[0]
            if accurate:
                real_ppn = ppn
                    
            else: #近似映射，查找oob
                actual = self.oob.lpn_to_ppn(lpn, source_page=ppn)
                if actual:
                    real_ppn = actual
                    oob_cached = False
                    for oob_page in self.last_oob_page:
                        if self.oob.lpn_to_ppn(lpn, source_page=oob_page):
                            oob_cached = True
                            break
                    if oob_cached:
                        pages_to_read += [actual]
                    else:
                        if actual == ppn:
                            pages_to_read += [ppn]
                        else:
                            pages_to_read += [ppn, actual]
                    if len(self.last_oob_page) >= 16:
                        self.last_oob_page.pop(0)
                    self.last_oob_page.append(ppn)
                # entry not exists; continue to search neighbor block (ppn is predicted to the wrong block)
                else:
                    pages_to_read += [ppn]
                    if ppn % self.conf.n_pages_per_block < self.conf.n_pages_per_block / 2.0:
                        ppn = int(ppn / self.conf.n_pages_per_block) * self.conf.n_pages_per_block - 1
                    else:
                        ppn = int(ppn / self.conf.n_pages_per_block + 1) * self.conf.n_pages_per_block
                    actual = self.oob.lpn_to_ppn(lpn, source_page=ppn)
                    try:
                        assert(actual)
                    except:
                        # if this assert fails, it is possible that prediction is out of oob range, but still in the same block
                        self.validation(lpn, None)
                    real_ppn = actual
                    if actual == ppn:
                        pages_to_read += [ppn]
                    else:
                        pages_to_read += [ppn, actual]

            self.validation(lpn, real_ppn)
        del results
        return real_ppn, pages_to_write, pages_to_read


    def update_pb(self, extents):
        assert(len(extents) == self.conf.n_pages_per_block * 8)
        extents = sorted(extents)
        entries = []
        pages_to_read = []
        pages_to_write = []
        victim_kv = []

        #next_free_block = self.bvc.next_free_block() #找到下一个分配物理页面的block及其ppn
        #next_free_ppn = self.conf.n_pages_per_block * next_free_block
        #self.pvb.validate_block(next_free_block)

        for lpn in extents:
            old_ppn = None
  
            if lpn not in self.page_based_mapping_table:
                continue

            old_ppn = self.page_based_mapping_table[lpn] # 找到历史ppn并将其置为无效
            if lpn in self.pb_mapping_table_cache:
                self.counter["pb_mapping_table_read_hit"] += 1
            else:
                self.counter["pb_mapping_table_read_miss"] += 1
                #yield self.env.process(self._read_ppns([0]))
                pages_to_read.append(0)


            if old_ppn:
                self.pvb.invalidate_page(old_ppn)
        
        for i in range(int(8)): # allocate 8 blocks
            next_free_block = self.bvc.next_free_block() #找到下一个分配物理页面的block及其ppn
            if next_free_block >= 33280:
                print("next_free_block", next_free_block)
            next_free_ppn = self.conf.n_pages_per_block * next_free_block # accloate 33280
            self.pvb.validate_block(next_free_block)
            for j, lpn in enumerate(extents[i*self.conf.n_pages_per_block:i*self.conf.n_pages_per_block+self.conf.n_pages_per_block]): # 分配ppn
                entry = (lpn, next_free_ppn + j)
                self.ppn_with_lpn[next_free_ppn + j] = lpn
                #print("over ppn 2", next_free_ppn + j)
                entries.append(entry) # 按顺序排好的(lpn, ppn)对
                self.page_based_mapping_table[lpn] = next_free_ppn + j
            #print("over ppn 2")
        
        #for i, lpn in enumerate(extents): # 分配ppn
            #entry = (lpn, next_free_ppn + i)
            #self.ppn_with_lpn[next_free_ppn + i] = lpn
            #entries.append(entry) # 按顺序排好的(lpn, ppn)对
            #self.page_based_mapping_table[lpn] = next_free_ppn + i
        
        for (lpn, ppn) in entries:
            if lpn in self.pb_mapping_table_cache:
                self.pb_mapping_table_cache[lpn] = ppn
                self.counter["pb_mapping_table_write_hit"] += 1
            else:
                self.pb_mapping_table_cache[lpn] = ppn
                self.counter["pb_mapping_table_write_miss"] += 1
        
        self.now_pb_mapping_table_cache_size = 8 * len(self.pb_mapping_table_cache) #bytes
        
        while self.now_pb_mapping_table_cache_size > self.pb_mapping_table_size :
            victim_key, victim_value = self.pb_mapping_table_cache.popitem(last=False)
            self.now_pb_mapping_table_cache_size -= 8
            #victim_kv.append(victim_key)
            #del self.pb_mapping_table_cache[victim_key]
            self.pb_mapping_table_storage[victim_key] = victim_value
            pages_to_read.append(1)
            pages_to_write = [0]
        
        
        return dict(entries), pages_to_read, pages_to_write


    '''
        @return Dict<lpn, ppn>
    '''
    def update(self, extents): #输入extents是分组后的lpa列表
        mappings = dict()
        pages_to_read = []
        pages_to_write = []
        #print("num", len(extents))
        for i in range(0, len(extents), self.conf.n_pages_per_block): # 每256为一组，一次截取一个256范围内的所有lpn
            submap, subpages_to_read, subpages_to_write = self.update_block(extents[i:i+self.conf.n_pages_per_block]) # 返回learn之后的结果
            # submap为按顺序排好的(lpn, ppn)对
            mappings.update(submap)
            pages_to_read.extend(subpages_to_read)
            pages_to_write.extend(subpages_to_write)


        return mappings, list(set(pages_to_read)), list(set(pages_to_write))

    '''
        @return Dict<lpn, ppn>
    '''
    def update_block(self, extents): # 输入的lpn范围在256之内 # 分配ppn, 更新pvb, 更新mapping table
        assert(len(extents) <= self.conf.n_pages_per_block)
        #if len(extents) != 256:
            #print("num", len(extents))
        entries = []
        pages_to_read = []
        pages_to_write = []

        extents = sorted(extents) #先排序

        # find the next free block
        next_free_block = self.bvc.next_free_block() #找到下一个分配物理页面的block及其ppn
        next_free_ppn = self.conf.n_pages_per_block * next_free_block


        # allocate flash pages
        for i, lpn in enumerate(extents):  # 排序后分配ppn
            entry = (lpn, next_free_ppn + i)
            self.ppn_with_lpn[next_free_ppn + i] = lpn
            entries.append(entry) # 按顺序排好的(lpn, ppn)对

        #TODO: additional flash reads; make this async; write to bitmap
        self.pvb.validate_block(next_free_block)

        for lpn in extents:
            old_ppn = None
            overheads = 0

            if not self.reference_mapping_table.get(lpn):
                continue

            old_ppn, pages_to_write, pages_to_read = self.lpn_to_ppn(lpn) # 找到历史ppn并将其置为无效

            if old_ppn:
                self.pvb.invalidate_page(old_ppn)

        mapping_pages_to_write, mapping_pages_to_read = self.mapping_table.update(entries, next_free_block) # 更新mapping table (学习并插入原来的组里)
        # 更新mapping table之后会涉及大小改变，如果过大，还是需要写入flash
        pages_to_write += mapping_pages_to_write
        pages_to_read += mapping_pages_to_read

        # update reference mapping table
        for (lpn, ppn) in entries:
            self.reference_mapping_table.set(lpn, ppn)

        # update oob
        # store the lpn of each ppn within [ppn - gamma - 1, ppn + gamma + 1]
        max_gamma = 16
        for i, (lpn, ppn) in enumerate(entries):
            upper = int(i + max_gamma + 2)
            lower = int(i - max_gamma - 1)
            self.oob.set_oob(ppn, entries[max(0,lower):upper])

        return dict(entries), pages_to_read, pages_to_write


    def validation(self, lpn, ppn):
        try:
            assert(ppn == self.reference_mapping_table.get(lpn))
        except:
            results, num_lookup, pages_to_write, pages_to_read = self.mapping_table.lookup(lpn, first=False)
            print("lpn:", lpn)
            print("reference ppn:",
                    self.reference_mapping_table.get_all(lpn))
            print("learned ppn:", ppn)
            print("all ppns in the tree:", results)
            for ppn, accurate, seg in results:
                if seg:
                    print("learned segment:", seg.full_str())
                print("oob data:", str(self.oob.oob_data[ppn]))
            exit(0)

# Learning-related components
class SimpleSegment():
    def __init__(self, k, b, x1, x2):
        self.b = b
        self.k = k
        self.x1 = x1
        self.x2 = x2

    def __str__(self):
        return "(%d, %.2f, %d, %d)" % (self.b, self.k, self.x1, self.x2)

    def __repr__(self):
        return str(self)

    def get_y(self, x):
        predict = int(round(x*self.k + self.b))
        return predict

    @staticmethod
    def intersection(s1, s2):
        p = (float(s2.b - s1.b) / (s1.k - s2.k),
             float(s1.k * s2.b - s2.k * s1.b) / (s1.k - s2.k))
        return p

    @staticmethod
    def is_above(pt, s):
        return pt[1] > s.k * pt[0] + s.b

    @staticmethod
    def is_below(pt, s):
        return pt[1] < s.k * pt[0] + s.b

    @staticmethod
    def get_upper_bound(pt, gamma):
        return (pt[0], pt[1] + gamma)

    @staticmethod
    def get_lower_bound(pt, gamma):
        return (pt[0], pt[1] - gamma)

    @staticmethod
    def frompoints(p1, p2):
        k = float(p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -k * p1[0] + p1[1]
        return SimpleSegment(k, b, p1[0], p2[0])


class Segment():
    FPR = 0.01
    PAGE_PER_BLOCK = 256
    BITMAP = True

    def __init__(self, k, b, x1, x2, points=None):
        self.b = b
        self.k = k
        self.x1 = x1
        self.x2 = x2
        self.accurate = True
        self.filter = None

        if points:
            self._points = points  # set of points only for verification purpose
            self.accurate, consecutive = self.check_properties(points)

            if not consecutive:
                if Segment.BITMAP:
                    self.filter = bitarray.bitarray(self.x2 - self.x1 + 1) 
                    self.filter.setall(0)
                    for pt in points:
                        self.filter[pt[0] - self.x1] = 1
                else:
                    self.filter = BloomFilter(len(points), Segment.FPR)
                    for pt in points:
                        self.filter.add(pt[0])
        
        if LPN_TO_DEBUG in zip(*points)[0]:
            log_msg("new seg", self)

    def __str__(self):
        return "%.4f, %d, [%d, %d], memory: %dB, accuracy: %s, bitmap: %s" \
            % (self.k, self.b, self.x1, self.x2, self.memory, self.accurate, self.filter)

    def __repr__(self):
        return str(self)
        return "(%d, %.4f, %d, %d, %s)" % (self.b, self.k, self.x1, self.x2, self.accurate)

    def full_str(self):
        return "(%d, %.4f, %d, %d, %s) " % (self.b, self.k, self.x1, self.x2, self.accurate) + str(self._points)

    def is_valid(self, x):
        if not (self.x1 <= x and x <= self.x2):
            return False
        if self.consecutive:
            return (x - self.x1) % self.rec_k == 0
        else:
            if Segment.BITMAP:
                return self.filter[x - self.x1]
            else:
                return self.filter.check(x)

    def get_y(self, x, check=True):
        if not check or self.is_valid(x):
            predict = int(round(x*self.k + self.b))
            # lowbound = self.blocknum * Segment.PAGE_PER_BLOCK
            # upbound = (self.blocknum + 1) * Segment.PAGE_PER_BLOCK - 1
            # return max(min(predict, upbound), lowbound)
            return predict
        return None

    # @return is_accruate, is_consecutive
    def check_properties(self, points):
        is_accruate, is_consecutive = True, True
        for pt in points:
            if self.get_y(pt[0], check=False) != pt[1]:
                is_accruate = False
            # if abs(self.get_y(pt[0], check=False) - pt[1]) > 5:
            #     print(self, self.get_y(pt[0], check=False), pt[1])
        if len(np.unique(np.diff(zip(*points)[0]))) > 1:
            is_consecutive = False

        return is_accruate, is_consecutive

    def overlaps(self, other):
        return min(self.x2, other.x2) - max(self.x1, other.x1) >= 0

    def overlaps_with_range(self, x1, x2):
        return min(self.x2, x2) - max(self.x1, x1) >= 0


    # check whether two segments can be put into the same level
    # if they can be put in the same level, return False
    # (here we assume other is older than self)
    # def conflict(self, other):
    #     if self.overlaps(other) < 0:
    #         return False
    #     # (self.x1 - other.x1) % self.rec_k == 0 is necessary -- consider (698, 700, 702) followed by (701, 703, 705)
    #     if self.k == other.k and self.accurate and other.accurate and not (other.x1 < self.x1 and self.x2 < other.x2) and (self.x1 - other.x1) % self.rec_k == 0:
    #         return False
    #     else:
    #         return True

    # @return new, old, same_level
    @staticmethod
    def merge(new, old):
        if not new.overlaps(old):
            return new, old, True

        if not new.mergable or not old.mergable:
            return new, old, False

        if new.consecutive and old.consecutive:
            if new.rec_k == old.rec_k and (new.x1 - old.x1) % old.rec_k == 0:
                if new.x1 <= old.x1 and old.x2 <= new.x2:
                    return new, None, True
                elif old.x1 < new.x1 and new.x2 < old.x2:
                    return new, old, False
                elif new.x1 <= old.x1:
                    old.x1 = new.x2 + new.rec_k
                    return new, old, True
                elif old.x1 < new.x1:
                    old.x2 = new.x1 - new.rec_k
                    return new, old, True
            # TODO: optimize for two-point segments
            # else:
            #     return new, old, False

        new, old = Segment.bitwise_merge(new, old)
        if not old:
            return new, None, True
        if not new.overlaps(old):
            return new, old, True
        else:
            return new, old, False


    @staticmethod
    def bitwise_merge(new, old):
        lo, hi = min(old.x1, new.x1), max(old.x2, new.x2)
        new_bm = bitarray.bitarray(hi-lo+1)
        old_bm = bitarray.bitarray(hi-lo+1)
        new_bm.setall(0)
        old_bm.setall(0)

        if new.consecutive:
            new_bm[new.x1-lo : new.x2-lo+1 : new.rec_k] = 1
        elif Segment.BITMAP:
            new_bm[new.x1-lo : new.x2-lo+1] = new.filter
        
        if old.consecutive:
            old_bm[old.x1-lo : old.x2-lo+1 : old.rec_k] = 1
        elif Segment.BITMAP:
            old_bm[old.x1-lo : old.x2-lo+1] = old.filter
        
        try:
            old_bm = old_bm & (~new_bm)
        except:
            print(lo, hi)
            print(old, new)
            print(old._points, new._points)
            print(old_bm, new_bm)
            exit(0)

        first_valid = old_bm.find(1)
        if first_valid == -1:
            return new, None
        last_valid = bitarray.util.rindex(old_bm, 1)
        old.x1 = first_valid + lo 
        old.x2 = last_valid + lo

        if not old.consecutive and Segment.BITMAP:
            old.filter = old_bm[first_valid : last_valid+1]
            assert(old.filter != None)

        # TODO: re-check accuracy and consecutive

        return new, old

    @property
    def consecutive(self):
        return not self.filter

    @property
    def mergable(self):
        return self.consecutive or Segment.BITMAP

    @property
    def length(self):
        return (self.x2-self.x1) // self.rec_k + 1

    @property
    def memory(self):
        if self.x1 == self.x2:
            return SUBLPN_BYTES + PPN_BYTES # + FLOAT16_BYTES + LENGTH_BYTES
        else:
            if self.consecutive:
                return SUBLPN_BYTES + PPN_BYTES + FLOAT16_BYTES + LENGTH_BYTES # 4+4+2+1
            else:
                if Segment.BITMAP:
                    # filter_size = len(self.filter) / 8.0
                    ones = len([e for e in self.filter if e])
                    non_consec_ones = len([i for i in range(len(self.filter)) if i > 0 and i < len(self.filter)-1 and self.filter[i] is not self.filter[i-1]])
                    # print(self.filter, [i for i in range(len(self.filter)) if i > 0 and i < len(self.filter)-1 and self.filter[i] is not self.filter[i-1]])
                    # zeros = len(self.filter) - ones
                    # sparse_encoding_size = min(ones, zeros) * 1 + LENGTH_BYTES
                    return SUBLPN_BYTES + PPN_BYTES + FLOAT16_BYTES + non_consec_ones * 1 + LENGTH_BYTES 
                    #return SUBLPN_BYTES + PPN_BYTES + FLOAT16_BYTES + min(filter_size, sparse_encoding_size)
                else:
                    return SUBLPN_BYTES + PPN_BYTES + FLOAT16_BYTES + round(self.filter.bit_array_size / 8.0)
    
    @property
    def rec_k(self):
        return int(round(1.0/self.k))
    @property
    def blocknum(self):
        mid = (self.x2 + self.x1) / 2.0
        predict = int(round(mid * self.k + self.b))
        return int(predict / Segment.PAGE_PER_BLOCK)


class PLR():
    FIRST = "first"
    SECOND = "second"
    READY = "ready"

    def __init__(self, gamma):
        self.gamma = gamma
        self.max_length = 256
        self.init()

    def init(self):
        # temp states to build one next segment
        self.segments = []
        self.s0 = None
        self.s1 = None
        self.rho_upper = None
        self.rho_lower = None
        self.sint = None
        self.state = PLR.FIRST
        self.points = []

    def learn(self, points):
        rejs = []
        count = 0
        size = len(points)
        for i, point in enumerate(points):
            seg, rej = self.process(point)
            if seg != None:
                self.segments.append(seg)
            if rej != None:
                rejs.append(rej)
    
        seg = self.build_segment()
        if seg != None:
            self.segments.append(seg)

        return self.segments

    def should_stop(self, point):
        if self.s1 is None:
            if point[0] > self.s0[0] + self.max_length:
                return True
        elif point[0] > self.s1[0] + self.max_length:
            return True
        return False

    def build_segment(self):
        if self.state == PLR.FIRST:
            seg = None
        elif self.state == PLR.SECOND:
            seg =  Segment(1, self.s0[1] - self.s0[0], self.s0[0], self.s0[0],
                           points=self.points)
        elif self.state == PLR.READY:
            avg_slope = np.float16((self.rho_lower.k + self.rho_upper.k) / 2.0)
            intercept = -self.sint[0] * avg_slope + self.sint[1]
            seg = Segment(avg_slope, intercept, self.s0[0], self.s1[0],
                           points=self.points)
    
        return seg

    def process(self, point):
        prev_segment = None
        if self.state == PLR.FIRST:
            self.s0 = point
            self.state = PLR.SECOND

        elif self.state == PLR.SECOND:
            if self.should_stop(point):
                prev_segment = self.build_segment()
                self.s0 = point
                self.state = PLR.SECOND
                self.points = []

            else:
                self.s1 = point
                self.state = PLR.READY
                self.rho_lower = SimpleSegment.frompoints(SimpleSegment.get_upper_bound(self.s0, self.gamma),
                                                    SimpleSegment.get_lower_bound(self.s1, self.gamma))
                self.rho_upper = SimpleSegment.frompoints(SimpleSegment.get_lower_bound(self.s0, self.gamma),
                                                    SimpleSegment.get_upper_bound(self.s1, self.gamma))
                self.sint = SimpleSegment.intersection(
                    self.rho_upper, self.rho_lower)
                self.state = PLR.READY

        elif self.state == PLR.READY:
            if not SimpleSegment.is_above(point, self.rho_lower) or not SimpleSegment.is_below(point, self.rho_upper) or self.should_stop(point):
                prev_segment = self.build_segment()
                self.s0 = point
                self.state = PLR.SECOND
                self.points = []

            else:
                self.s1 = point
                s_upper = SimpleSegment.get_upper_bound(point, self.gamma)
                s_lower = SimpleSegment.get_lower_bound(point, self.gamma)
                if SimpleSegment.is_below(s_upper, self.rho_upper):
                    self.rho_upper = SimpleSegment.frompoints(self.sint, s_upper)
                if SimpleSegment.is_above(s_lower, self.rho_lower):
                    self.rho_lower = SimpleSegment.frompoints(self.sint, s_lower)

        self.points.append(point)

        return (prev_segment, None)


# bisect wrapper
class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


# 一个组，即一个log-structured结构
class LogPLR():
    def __init__(self, gamma, frame_no):
        self.plr = PLR(gamma)
        # one run is one level of segments with non-overlapping intervals
        self.runs = [] # run == level
        self.frame_no = frame_no
        # mapping from block to segments
        # self.block_map = defaultdict(list)
    
    def update(self, entries, blocknum):
        # make sure no same LPNs exist in the entries
        sorted_entries = sorted(entries)
        # make sure no same 'x1's exist in the new_segments

        self.plr.init()
        new_segments = self.plr.learn(sorted_entries)
        new_segments = sorted(new_segments, key=lambda x: x.x1)

        # self.block_map[blocknum].extend(new_segments)
        # make sure no overlap at each level
        self.add_segments(0, new_segments, recursive=False)

        return []

    def merge(self, old_plr):
        assert(self.frame_no == old_plr.frame_no)
        self.runs.extend(old_plr.runs)

    # bottleneck # 在一个group里的log-structured里查找lpa
    def lookup(self, LPA, first=True): ## frame.lookup()
        empty_levels = []
        results = []
        lookup = 0
        for level, run in enumerate(self.runs): # 遍历每个level
            if len(run) == 0:
                empty_levels.append(level)
                continue
            lookup += 1
            index = bisect_left(KeyWrapper(run, key=lambda seg: seg.x1), LPA)
            if index == 0 or (index < len(run) and run[index].x1 == LPA):
                seg = run[index]
            else:
                seg = run[index - 1]
            PPA = seg.get_y(LPA)
            if PPA:
                results += [(PPA, seg.accurate, seg)]
                if first:
                    break

        for level in sorted(empty_levels, reverse=True):
            del self.runs[level]
        
        return results, lookup, [], []

    def lookup_range(self, start, end):
        results = defaultdict(list)
        for level, run in enumerate(self.runs): #遍历每一层
            if len(run) == 0:
                continue
            index = bisect_left(KeyWrapper(run, key=lambda seg: seg.x1), start) # 查找seg.x1是否在start至len(run)里
            if index == 0 or (index < len(run) and run[index].x1 == start):
                pass
            else:
                index = index - 1

            while index < len(run):
                seg = run[index]
                if seg.overlaps_with_range(start, end):
                    results[level].append(seg)
                    index += 1
                else:
                    break

        return results # 返回具有重叠lpa range的segments
            

    # recursively add segments to each level
    # bottleneck
    def add_segments(self, level, segments, recursive=True):
        while len(self.runs) <= level:
            self.runs.append([])

        run = self.runs[level]
        conflicts = []
        for new_seg in segments:
            if new_seg.get_y(LPN_TO_DEBUG):
                log_msg("%s added to run %d" % (new_seg, level))
            if len(run) == 0:
                run.append(new_seg)
                continue

            index = bisect_left(KeyWrapper(
                run, key=lambda seg: seg.x1), new_seg.x1)
            run.insert(index, new_seg)
            overlaps = []
            if index != 0:
                overlaps.append((index-1, run[index-1]))
            for i in range(index+1, len(run)):
                if run[i].x1 > new_seg.x2:
                    break
                overlaps.append((i, run[i]))

            indices_to_delete = []
            for index, old_seg in overlaps:
                to_print = old_seg.get_y(LPN_TO_DEBUG)
                if to_print:
                    log_msg("%s tries to merge with %s" % (new_seg, old_seg))
                new_seg, old_seg, same_level = Segment.merge(new_seg, old_seg)

                if not old_seg:
                    indices_to_delete.append(index)
                    if to_print:
                        log_msg("%s removed old seg" % (new_seg))
                elif not same_level:
                    conflicts.append(old_seg)
                    indices_to_delete.append(index)
                    if to_print:
                        log_msg("%s -> %s" % (new_seg, old_seg))
            for index in sorted(indices_to_delete, reverse=True):
                if run[index].get_y(LPN_TO_DEBUG):
                    log_msg("removed old seg", (run[index]))
                del run[index]

        if recursive:
            if len(conflicts) > 0:
                self.add_segments(level+1, conflicts)
        else:
            if len(conflicts) > 0:
                self.runs.insert(level+1, conflicts)
            
    def __str__(self):
        repr = ""
        for level in range(len(self.runs)):
            repr += "== level %d ==\n %s \n" % (level, str(self.runs[level]))
        return repr

    @property
    def segments(self):
        return [seg for run in self.runs for seg in run]

    @property
    def memory(self):
        return sum([seg.memory for i, run in enumerate(self.runs) for seg in run]) + LPN_BYTES

    @property
    def levels(self):
        return len(self.runs)

    # now we only use compact
    def gc(self, blocknum):
        return 
        for seg in self.block_map[blocknum]:
            for run in reversed(self.runs):
                index = bisect_left(KeyWrapper(run, key=lambda _: _.x1), seg.x1)
                if index < len(run) and seg == run[index]:
                    if seg.get_y(LPN_TO_DEBUG):
                        log_msg("gc removed old seg", (seg))
                    run.remove(seg)
                    break
 
        self.block_map[blocknum] = []

    def promote(self): # 
        if len(self.runs) == 0:
            return

        if self.levels <= 10: # level数大于10
            return

        layers = self.runs[:]
        for i in range(1,len(layers)):
            lower_layer = layers[i]
            for old_seg in reversed(lower_layer):
                promoted_layer = i
                promoted_index = None
                for j in reversed(range(0,i)):
                    upper_layer = layers[j]
                    index = bisect_left(KeyWrapper(upper_layer, key=lambda seg: seg.x1), old_seg.x1)
                    overlaps = False
                    for k in range(max(0, index-1), len(upper_layer)):
                        if upper_layer[k].x1 > old_seg.x2:
                            break
                        if upper_layer[k].overlaps(old_seg):
                            overlaps = True
                            break
                    if overlaps:
                        break
                    else:
                        promoted_layer = j
                        promoted_index = index
                
                if promoted_layer < i:
                    # if self.frame_no == 3075:
                    #     log_msg("Promote %s to level %d" % (old_seg, promoted_layer))
                    layers[promoted_layer].insert(promoted_index, old_seg)
                    lower_layer.remove(old_seg)
        
        self.runs = [run for run in self.runs if len(run) != 0]
                  
    def compact(self, promote=False):  # 组内compaction
        if len(self.runs) == 0: # run == level
            return

        # relearn_dict = dict()
        layers = self.runs[:1] # 第一层
        for layer in layers:
            for seg in layer: # 遍历第一层每个segment
                self.compact_range(seg.x1, seg.x2)
                # relearn_dict.update(relearn)

        self.runs = [run for run in self.runs if len(run) != 0]
        if promote:
            self.promote()

        return#len(self.segments), relearn_dict

    def compact_range(self, start, end): # 遍历第一层每一个segment
        results = self.lookup_range(start, end) # 找出重叠的segments集合
        # relearn = dict()
        for upper_layer, new_segs in results.items():
            for lower_layer, old_segs in results.items():
                if upper_layer < lower_layer:
                    for new_seg in new_segs:
                        for old_seg in old_segs:
                            # if old_seg not in relearn and old_seg.x1 < new_seg.x1 and new_seg.x2 < old_seg.x2:
                            #     if not old_seg.consecutive:
                            #         relearn[old_seg] = sum(old_seg.filter) - 2
                            new_seg, updated_old_seg, same_level = Segment.merge(new_seg, old_seg)
                            if not updated_old_seg:
                                self.runs[lower_layer].remove(old_seg)
                                results[lower_layer].remove(old_seg)
        # return dict() #relearn

# Distribute the mapping entries into LPN ranges
# Each LogPLR is responsible for one range
# 所有组的集合，一个LRUCache
# dftl, sftl, leaftl都是LRUcache，只不过里面的基本粒度不一样，leaFTL的粒度是一个组，dftl的粒度是1-1的mapping table (8 byte)
class FrameLogPLR:
    ON_FLASH, CLEAN, DIRTY = "ON_FLASH", "CLEAN", "DIRTY"
    def __init__(self, conf, metadata, counter, gamma, max_size=1*1024**2, frame_length=256):
        global SUBLPN_BYTES
        SUBLPN_BYTES = 1
        self.conf = conf
        self.metadata = metadata
        self.counter = counter
        self.gamma = gamma
        self.frame_length = frame_length
        self.frames = LRUCache()
        self.leaftl_hit = 0
        self.leaftl_size = 50*MB # 584056832 bytes = 557 MB
        log_msg("mapping cache size", self.leaftl_size)

        # internal_type = "sftl"
        if self.conf['internal_ftl_type'] == "sftl":
            self.type = "sftl"
            self.frame_length = 1024
        elif self.conf['internal_ftl_type'] == "dftldes":
            self.type = "dftldes"
            self.frame_length = 1024
        else:
            self.type = "learnedftl"
            self.frame_length = 256

        self.GTD = dict()
        self.current_trans_block = None
        self.current_trans_page_offset = 0
        self.frame_on_flash = dict()
        self.memory_counter = dict()
        self.total_memory = 0
        self.hits = 0
        self.misses = 0
        self.dirty = dict()
        self.compact_time = 0
        self.compact_time_all = 0
        self.compact_change_time = 0
        self.compact_change_time_all = 0

    def create_frame(self, frame_no): # 在这里区分
        if self.type == "learnedftl":
            return LogPLR(self.gamma, frame_no)
        
        elif self.type == "sftl":
            return SFTLPage(frame_no, self.frame_length)

        elif self.type == "dftldes":
            return DFTLPage(frame_no)
            (frame_no)
        
        else:
            raise NotImplementedError

    @staticmethod
    def split_into_frame(frame_length, entries):
        split_results = defaultdict(list)
        for lpn, ppn in entries:
            split_results[lpn // frame_length].append((lpn, ppn)) #添加到对应组里
        return split_results

    def update(self, entries, blocknum):
        pages_to_write, pages_to_read = [], []

        split_entries = FrameLogPLR.split_into_frame(self.frame_length, entries)
        
        frame_nos = []
        for frame_no, entries in split_entries.items():
            frame_nos += [frame_no]

            if frame_no not in self.frames:
                self.frames[frame_no] = self.create_frame(frame_no)
                self.counter["mapping_table_write_miss"] += 1
            else:
                self.counter["mapping_table_write_hit"] += 1
            
            self.frames[frame_no].update(entries, blocknum) # 在这里learn

            if self.frames[frame_no].levels > 0:
                self.frames[frame_no].promote()

            self.dirty[frame_no] = True # 在mapping table cache里更改 但是并没有写进flush

            #print(self.frames[frame_no].memory)
            self.change_size_of_frame(frame_no, self.frames[frame_no].memory)
        
        if self.should_flush():
            pages_to_write, pages_to_read = self.flush()

        return pages_to_write, pages_to_read

    def lookup(self, lpn, first=True): # 输入lpn是原始lpn
        should_print = False
        frame_no = lpn // self.frame_length  #组号
        results = []
        pages_to_write, pages_to_read = [], []

        if frame_no in self.frames:  # frame是一个LRUCache # 每个group是一个log-structured结构，group不可分割，在DRAM里以LRU策略存储
            # self.hits += 1
            frame = self.frames[frame_no] # 取出group
            self.frames.move_to_head(frame_no, frame) # LRU cache
            # log_msg("Move to head", frame_no)
            results, lookup, _, _ = frame.lookup(lpn, first) # 去group的log-structue里查找
  
        if len(results) != 0: # 在DRAM里
            self.counter["mapping_table_read_hit"] += 1
        else: # 需要去flash取出这个组
            self.leaftl_hit += 1
            self.counter["mapping_table_read_miss"] += 1
            frame = self.frame_on_flash[frame_no]
            blocknum = self.GTD[frame_no]
            results, lookup, _, _ = frame.lookup(lpn, first)
            pages_to_read = [blocknum]

            if frame_no in self.frames:
                self.frames[frame_no].merge(frame)
                del self.frame_on_flash[frame_no]
            else:
                self.dirty[frame_no] = False
                self.frames[frame_no] = frame
                del self.frame_on_flash[frame_no]
                
            self.frames.move_to_head(frame_no, self.frames[frame_no])
            self.change_size_of_frame(frame_no, self.frames[frame_no].memory)

        if self.memory > self.leaftl_size:
            should_print = True

        if self.should_flush(): # mapping table占的空间过大
            mapping_pages_to_write, mapping_pages_to_read = self.flush()
            pages_to_read += mapping_pages_to_read
            pages_to_write += mapping_pages_to_write


        # results是查找到的ppa，lookup是查找的次数
        return results, lookup, pages_to_write, pages_to_read

    # gc is currently replaced with compaction
    def gc(self, blocknum):
        pass

    def compact(self, promote=False, frame_nos=None):
        if not frame_nos:
            for frame_no, frame in self.frames.items(): # 遍历所有log-structured组
                frame.compact(promote=promote)

                self.change_size_of_frame(frame_no, frame.memory) 
        else:
            for frame_no in frame_nos:
                frame = self.frames[frame_no]
                frame.compact(promote=promote)
                self.change_size_of_frame(frame_no, frame.memory) 

    def promote(self):
        for frame in self.frames.values():
            frame.promote()


    def should_flush(self):
        if self.memory > self.leaftl_size:
            return True
        else:
            return False

    def allocate_ppn_for_frame(self, frame_no):
        if not self.current_trans_block or self.current_trans_page_offset == 256:
            self.current_trans_block = 0
            self.current_trans_page_offset = 0
            

        next_free_ppn = self.conf.n_pages_per_block * self.current_trans_block + self.current_trans_page_offset
        #self.current_trans_page_offset += 1

        if frame_no not in self.GTD:
            old_ppn = None
            self.GTD[frame_no] = next_free_ppn
            new_ppn = self.GTD[frame_no]
            
        else:
            old_ppn = self.GTD[frame_no]
            self.GTD[frame_no] = next_free_ppn
            new_ppn = self.GTD[frame_no]

        return new_ppn, old_ppn
        
    def flush(self):
        evicted_frames = []
        pages_to_read = []
        pages_to_write = []


        original_memory = self.memory
        while original_memory > self.leaftl_size:
            frame_no, evict_frame = self.frames.popitem(last=False)
            # log_msg(frame_no, "evicted")
            freed_mem = self.memory_counter[frame_no]
            original_memory -= freed_mem
            self.change_size_of_frame(frame_no, 0)
            evicted_frames.append(frame_no)

            new_ppn, old_ppn = self.allocate_ppn_for_frame(frame_no)
            if self.dirty[frame_no]:
                self.counter["flush mapping table"] += 1
            self.dirty[frame_no] = False
            pages_to_write.append(new_ppn)

            if frame_no in self.frame_on_flash:
                # log_msg(frame_no, "merged with flash")
                old_frame = self.frame_on_flash[frame_no]
                pages_to_read += [old_ppn]
                evict_frame.merge(old_frame)
            self.frame_on_flash[frame_no] = evict_frame

        return pages_to_write, list(set(pages_to_read))

    def change_size_of_frame(self, frame_no, new_mem):
        old_mem = 0
        if frame_no in self.memory_counter:
            old_mem = self.memory_counter[frame_no]
        self.memory_counter[frame_no] = new_mem
        self.total_memory += (new_mem - old_mem)
  
    # TODO: bottleneck
    @property
    def memory(self):
        # assert(self.total_memory == sum([mem for frame_no, mem in self.memory_counter.items()]))
        return self.total_memory
        # return sum([mem for frame_no, mem in self.memory_counter.items()])

    @property
    def segments(self):
        return [seg for frame in self.frames.values() for seg in frame.segments]

    @property
    def levels(self):
        if len(self.frames) == 0:
            return 0
        return max([frame.levels for frame in self.frames.values()])

    @property
    def dist_levels(self):
        if len(self.frames) == 0:
            return 0, 0
        dist = [frame.levels for frame in self.frames.values() if frame.levels != 0]
        return dist
       
# reference_mapping_table，存储历史版本，具体作用需要再详细看
# 是1-1的设计?
class PFTL(object):
    def __init__(self):
        # store update history for verification purpose
        self.mapping_table = defaultdict(list)

    def set(self, lpn, ppn):
        self.mapping_table[lpn].append(ppn)
            
    def get(self, lpn):
        # force check; since we are using defaultdict we don't want to create empty entry
        if lpn not in self.mapping_table:
            return None
        ppns = self.mapping_table[lpn]
        if len(ppns) > 0:
            return ppns[-1]
    
    def delete(self, lpn):
        del self.mapping_table[lpn]

    def get_all(self, lpn):
        if lpn not in self.mapping_table:
            return None
        return self.mapping_table[lpn]

    @property
    def memory(self):
        return len(self.mapping_table) * (PPN_BYTES + LPN_BYTES)

def split_ext(extent):
    if extent.lpn_count == 0:
        return None

    exts = []
    page_num = 8 * 1024 * 1024
    for lpn in extent.lpn_iter():
        cur_ext = Extent(lpn_start=lpn % page_num, lpn_count=1, timestamp = extent.timestamp)
        exts.append(cur_ext)

    return exts

# no need to dump the entire timeline if storage space is limited
def write_timeline(conf, recorder, op_id, op, arg, start_time, end_time):
    return
    recorder.write_file('timeline.txt',
            op_id = op_id, op = op, arg = arg,
            start_time = start_time, end_time = end_time)