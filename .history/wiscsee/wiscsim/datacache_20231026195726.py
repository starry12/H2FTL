#!/usr/bin/env python2
from collections import OrderedDict, defaultdict
import numpy as np
from wiscsim.utils import *


class LRUCache(OrderedDict):
    def __init__(self):
        super(LRUCache, self).__init__()

    def append(self, key):
        self[key] = ""

    # FIXME: bug popitem
    def evict(self, n):
        evicted = []
        for _ in range(n):
            if len(self) == 0:
                return None
            lpn, state = self.popitem(last=False)
            evicted.append((lpn, state))
        return evicted

    def evict_one(self):
        evicted = self.evict(1)
        return evicted[0]

    def evict_lpn(self, lpn):  # evict function
        state = self[lpn]
        del self[lpn]
        return (lpn, state)

    def _write(self, lpn, state=""): # LRU_write
        self[lpn] = state
        #self._end(lpn)
        #self.move_to_end(lpn, last = True)
    
    def _read(self, lpn):
        self.move_to_head(lpn, self[lpn])
        #self._end(lpn)
        #self.move_to_end(lpn, last = True)
        return self[lpn]
    
    def _end(self, lpn):
        self.move_to_end(lpn, last = True)

    def peak(self):
        return next(self.iteritems())

    def peak_n(self, n):
        return dict([next(self.iteritems()) for i in range(n)])

    def move_to_head(self, key, value, dict_setitem=dict.__setitem__):
        del self[key]
        self[key] = value
        # print(dir(self))
        # root = self._OrderedDict__root
        # first = root[1]

        # if key in self:
        #     if first[2] != key:
        #         link = self._OrderedDict__map[key]
        #         link_prev, link_next, _ = link
        #         link_prev[1] = link_next
        #         link_next[0] = link_prev
        #         link[0] = root
        #         link[1] = first
        #         root[1] = first[0] = link
        # else:
        #     root[1] = first[0] = self._OrderedDict__map[key] = [root, first, key]
        # dict_setitem(self, key, value)


"""
    self.cache = {LPN : priority in {True or False}}
    self.high = # of entries with priority
"""
class DataCache(object):
    def __init__(self, size, slot_size, method="LRU", priority=False, threshold=0.2):
        
        assert(method in ["LRU","FIFO","LFU","LIFO","MRU"])
        method2cache = {"LRU": LRUCache()}
        self.cache = method2cache[method]
        self.slot_size = slot_size
        self.capacity = max(1, self.bytes2slots(size))
        self.priority = priority
        self.high = 0.
        self.threshold = threshold

    def bytes2slots(self, size):
        return max(int(size / self.slot_size), 0)
        
    def read(self, lpn):
        if lpn not in self.cache:
            self._write(lpn)
            return False
        else:
            self.cache._read(lpn)
            # self.cache.move_to_head(lpn, self.cache[lpn])
            return True

    def invalidate(self, lpn):
        if lpn in self.cache:
            # priority = self.cache[lpn]
            del self.cache[lpn]
            # if priority:
            #     self.high -= 1

    def set_priority(self, lpn, priority):
        old_priority = self.cache[lpn]
        if lpn in self.cache:
            self.cache[lpn] = priority
            if not old_priority and priority:
                self.high += 1
            if old_priority and not priority:
                self.high -= 1


    def _write(self, lpn):
        # self.cache[lpn] = False
        # self.cache.move_to_head(lpn, self.cache[lpn])
        if len(self.cache) >= self.capacity:
            self.evict(1)
        self.cache._write(lpn)

    def evict(self, n):
        assert(n <= self.capacity)
        self.cache.evict(n)
        # for _ in range(n):
        #     # guarantee to evict low priority
        #     # if self.priority and self.high <= self.threshold * self.capacity:
        #     #     for lpn, priority in reversed(self.cache.items()):
        #     #         if not priority:
        #     #             del self.cache[lpn]
        #     #             break
        #     # else:
        #     lpn, priority = self.cache.popitem(last=False)
                # if priority:
                #     self.high -= 1

    def resize(self, size):
        assert(size >= 0)
        old_capacity = self.capacity
        self.capacity = self.bytes2slots(size)
        if self.capacity < old_capacity:
            self.evict(old_capacity - self.capacity)


class WriteBuffer():
    UNASSIGNED, ASSIGNED = "UNASSIGNED", "ASSIGNED"
    def __init__(self, flush_size, size, filtering = 1.0, assign_ppn=True):
        assert(flush_size <= size)
        self.assign_ppn = assign_ppn
        self.flush_size = flush_size
        self.size = size
        self.assigned = LRUCache()
        self.unassigned = LRUCache()
        self.mapping = dict()
        self.counter = 0
        self.filtering = filtering

    def peek(self, lpn, move=False):
        if lpn in self.assigned:
            if move:
                self.assigned.move_to_head(lpn, self.assigned[lpn])
            return WriteBuffer.ASSIGNED
        elif lpn in self.unassigned:
            if move:
                self.unassigned.move_to_head(lpn, self.unassigned[lpn])
            return WriteBuffer.UNASSIGNED 
        else:
            return None 

    def read(self, lpn):
        return self.peek(lpn, move=True)


    def write(self, lpn):
        if not self.peek(lpn):
            self.unassigned[lpn] = ""
        else:
            self.counter += 1
            #print("hit", self.counter)

        if self.length > self.size:
            writeback, _ = self.assigned.popitem(last=False)
            if self.assign_ppn:
                ppn = self.mapping[writeback]
                del self.mapping[writeback]
                return writeback, ppn
            else:
                return writeback, None

        else:
            return (None, None)
        

    def should_assign_page(self):
        return len(self.unassigned) >= self.flush_size

    @staticmethod
    def split_into_frame(frame_length, entries):
        split_results = defaultdict(list)
        for lpn, ppn in entries.items():
            split_results[lpn // frame_length].append((lpn, ppn))
        return split_results

    def flush_unassigned(self):
        assert(len(self.unassigned) == self.flush_size)
        split_results = self.split_into_frame(256, self.unassigned)
        per_frame_count = dict(sorted([(k, len(v)) for k,v in split_results.items()], key=lambda x:x[1], reverse=True))
        cumsum = np.cumsum(per_frame_count.values())
        index = np.argwhere(cumsum<=self.flush_size*self.filtering)[-1][0]
        lpa_to_flush = sorted(filter(lambda x: x // 256 in per_frame_count.keys()[:index+1], self.unassigned))

        for lpa in lpa_to_flush:
            self.assigned[lpa] = ""

        for lpn in lpa_to_flush:
            del self.unassigned[lpn]

        # buffer = sorted(self.unassigned)
        # self.assigned.update(self.unassigned)
        # self.unassigned = LRUCache()
        return lpa_to_flush

    def update_assigned(self, mappings):
        assert(self.assign_ppn)
        for lpn, ppn in mappings.items():
            assert(lpn in self.assigned)
            self.mapping[lpn] = ppn

    @property
    def length(self):
        return len(self.assigned) + len(self.unassigned)


class SimpleWriteBuffer():
    def __init__(self, size):
        # assert(size <= max_size)
        # self.max_size = max(size, max_size)
        self.size = size
        self.buffer = []
        self.counter = 0

    def read(self, lpn):
        return lpn in self.buffer

    def write(self, lpn):
        if lpn not in self.buffer:
            self.buffer.append(lpn)
        else:
            self.counter += 1
            #print("hit", self.counter)

    def should_flush(self):
        return self.length >= self.size

    def flush(self):
        written_buffer = self.buffer
        self.buffer = []
        return sorted(written_buffer)

    @property
    def length(self):
        return len(self.buffer)


class RWCache_hot(object):
    def __init__(self, size, slot_size, flush_size, filter_ratio, preconditioning=0.2, method="LRU"):
        # self.rw_cache = RWCache(self.conf['cache_size'], self.conf.page_size, 8*MB, 7.0/8.0 = 0.875)

        assert(method in ["LRU","FIFO","LFU","LIFO","MRU"])
        self.slot_size = slot_size # self.conf.page_size
        if flush_size > size: # if 8MB > cache_size  
            flush_size = size
            filter_ratio = 1.0
            log_msg("Adjusting flush size to become the data cache size")
        self.capacity = max(int(size / self.slot_size), 1)  # cache_size / page_size
        assert(flush_size <= size)
        self.flush_size = max(int(flush_size / self.slot_size), 1) 
        #print("flush_size",self.flush_size)
        self.flush_size_hot = self.flush_size
        #self.flush_size_cold = self.flush_size

        self.cache = self.method2cache(method)
        self.clean = self.method2cache(method)
        self.assigned = self.method2cache(method)
        self.unassigned_hot = self.method2cache(method) # LRU
        self.evict_hot = self.method2cache(method)
        self.overall_data_cache_size = int(126*MB / self.slot_size)
        #self.unassigned_cold = self.method2cache(method)
        self.dirty_page_mapping = dict()
        self.filter_ratio = filter_ratio # dftl: 0.875  sftl/leaftl: 1.0
        self.inplace_update = 0
        self.counter = defaultdict(float)

        
        for i in range(self.capacity):
            dummy_lpn = "dummy"+str(i)
            if i <= int(preconditioning*self.capacity):
                self.cache._write(dummy_lpn, state="A") # self["dummy"+str(i)] = "A"
                self.assigned._write(dummy_lpn) # self["dummy"+str(i)] = ""
            else:
                self.cache._write(dummy_lpn, state="C") # self["dummy"+str(i)] = "C"
                self.clean._write(dummy_lpn) # self["dummy"+str(i)] = ""
            # TODO: change 0 to a random ppn in the address space and keep them striped across different channels.
            self.dirty_page_mapping[dummy_lpn] = 0

    def method2cache(self, method):
        if method == "LRU":
            return LRUCache()
        
    def read(self, lpn): # rw_cache.read(ext.lpn_start)
        writeback, wb_ppn = False, None
        if lpn not in self.cache:
            self.counter["read misses"] += 1
            self.cache._write(lpn, state="C") # LRU --- self[lpn] = "C"
            self.clean._write(lpn) # self[lpn] = ""
            writeback, wb_ppn = self.evict(state_preference="C")
            hit = False
        else: 
            self.counter["read hits"] += 1
            state = self.cache._read(lpn)
            if state == "H":
                self.unassigned_hot._read(lpn)
            if state == "A":
                self.assigned._read(lpn)
            if state == "C":
                self.clean._read(lpn)
            hit = True

        if writeback:
            self.counter["writebacks"] += 1
        
        return hit, writeback, wb_ppn
    

    def read_evict_hot(self, lpn):
        if lpn in self.evict_hot:
            hit = True
        else:
            hit = False
        return hit
    

    def read_and_delete(self, lpn):
        wb_ppn = None
        if lpn not in self.cache: 
            return
        else:
            state = self.cache._read(lpn)
            if state == "A":
                self.assigned.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_hot._write(lpn)
            if state == "H":
                return
            if state == "C":
                self.clean.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_hot._write(lpn)
        return wb_ppn
    
    def delete(self, lpn):
        if lpn in self.cache:
            state = self.cache._read(lpn)
            if state == "A":
                return
            if state == "H":
                self.unassigned_hot.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_hot._write(lpn)
                self.inplace_update += 1
            if state == "C":
                self.clean.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_hot._write(lpn)
                


    def traversal_unassigned(self):
        unassigned_lpn_num = len(self.unassigned_hot) + len(self.assigned)
        inplace_update_num = self.inplace_update
        return unassigned_lpn_num, inplace_update_num

    def write(self, lpn, hotness): # hotness == hot
        hit, writeback, wb_ppn = False, False, None
        if lpn not in self.cache:
            self.counter["write misses"] += 1
            self.cache._write(lpn, state="H")
            self.unassigned_hot._write(lpn)
            writeback, wb_ppn = self.evict(state_preference="C")
        else: # in cache
            hit = True
            self.counter["write hits"] += 1
            state = self.cache._read(lpn)
            if state == "H":
                self.unassigned_hot._read(lpn)
                self.inplace_update += 1
            if state == "A":
                self.assigned._read(lpn)
                self.inplace_update += 1
            if state == "C":
                self.clean.evict_lpn(lpn)
                self.cache._write(lpn, state="H")
                self.unassigned_hot._write(lpn)

        if writeback:
            self.counter["writebacks"] += 1
        
        return hit, writeback, wb_ppn
    
    def evict(self, state_preference): # hot
        writeback, wb_ppn = False, None
        if self.should_evict():
            lpn, state = self.cache.peak()
            if state == "A":
                self.cache.evict_one()
                lpn, state = self.assigned.evict_lpn(lpn) 
                self.evict_hot._write(lpn)
                writeback, wb_ppn = True, self.dirty_page_mapping[lpn]
                del self.dirty_page_mapping[lpn]
            elif state == "C": 
                self.cache.evict_one()
                lpn, state = self.clean.evict_lpn(lpn)
                self.evict_hot._write(lpn)
            
            ## we still can hit U here if the size of unassigned is smaller than flush_size
            elif state == "H": # state_preference="C"
                if len(self.assigned) == 0 and len(self.clean) == 0:
                    return writeback, wb_ppn

                if state_preference == "A" and len(self.assigned) == 0:
                    state_preference = "C"
                elif state_preference == "C" and len(self.clean) == 0:
                    state_preference = "A"
                                
                if state_preference == "A":
                    lpn, _ = self.assigned.evict_one()
                    _, state = self.cache.evict_lpn(lpn)
                    self.evict_hot._write(lpn)
                    assert(state == state_preference)
                    writeback, wb_ppn = True, self.dirty_page_mapping[lpn]
                    del self.dirty_page_mapping[lpn]
                elif state_preference == "C":
                    lpn, _ = self.clean.evict_one()
                    _, state = self.cache.evict_lpn(lpn)
                    self.evict_hot._write(lpn)
                    assert(state == state_preference)
        
        if self.capacity + len(self.evict_hot) >= self.overall_data_cache_size:
            while self.capacity + len(self.evict_hot) >= self.overall_data_cache_size:
                self.evict_hot.evict_one()
        
        return writeback, wb_ppn

    
    def update_assigned(self, mappings):
        for lpn, ppn in mappings.items():
            assert(lpn in self.assigned)
            self.dirty_page_mapping[lpn] = ppn 

    def select_unassigned(self, hotness): # hotness == 'hot'
        frame_length = 256
        split_results = defaultdict(list)
        candidates = []
        if hotness == 'hot':
            for i, (lpn, _) in enumerate(self.unassigned_hot.items()):
                split_results[lpn // frame_length].append((lpn, _))
                candidates.append(lpn)
                if i >= self.flush_size_hot - 1:
                    break

        per_frame_count = dict(sorted([(k, len(v)) for k,v in split_results.items()], key=lambda x:x[1], reverse=True))
        cumsum = np.cumsum(per_frame_count.values())

        ## select index as the last logical group to satisfy the flush size requirement
        index = np.argwhere(cumsum<=self.flush_size_hot*self.filter_ratio)[-1][0]
        #log_msg("index %d" % index)

        lpa_to_flush = sorted(filter(lambda x: x // frame_length in per_frame_count.keys()[:index+1], candidates)) 

        return lpa_to_flush


    def flush_unassigned(self, hotness):
        if hotness == 'hot':
            lpn_to_flush = self.select_unassigned(hotness = 'hot')
            for lpn in lpn_to_flush:
                self.assigned[lpn] = "" 
                del self.unassigned_hot[lpn]
                self.cache[lpn] = "A" # assigned 
                
        return lpn_to_flush
    
    def should_assign_page_hot(self):
        return (self.cache.peak()[1] == "H") and len(self.cache) >= self.capacity and len(self.unassigned_hot) >= self.flush_size_hot

    def should_evict(self):
        return len(self.cache) > self.capacity
    
    def remaining_size(self):
        return self.capacity - len(self.cache)  # cache_size / page_size
    
    def change_capacity(self, add_or_reduce, size):
        if add_or_reduce == True:
            self.capacity += size / self.slot_size
        else:
            self.capacity -= size / self.slot_size


class RWCache_cold(object):
    def __init__(self, size, slot_size, flush_size, filter_ratio, preconditioning=0.2, method="LRU"): 
        # self.rw_cache = RWCache(self.conf['cache_size'], self.conf.page_size, 8*MB, 7.0/8.0 = 0.875)

        assert(method in ["LRU","FIFO","LFU","LIFO","MRU"])
        self.slot_size = slot_size # self.conf.page_size
        if flush_size > size: # if 8MB > cache_size  
            flush_size = size 
            filter_ratio = 1.0
            log_msg("Adjusting flush size to become the data cache size")
        self.capacity = max(int(size / self.slot_size), 1)  # cache_size / page_size  
        assert(flush_size <= size)
        self.flush_size = max(int(flush_size / self.slot_size), 1)
        self.flush_size_cold = self.flush_size

        self.cache = self.method2cache(method)
        self.assigned = self.method2cache(method) # "A"
        self.unassigned_cold = self.method2cache(method) # "CL"
        self.evict_cold = self.method2cache(method)
        self.overall_data_cache_size = int(126*MB / self.slot_size)
        self.dirty_page_mapping = dict()
        self.filter_ratio = filter_ratio # dftl: 0.875  sftl/leaftl: 1.0
        self.inplace_update = 0

        self.counter = defaultdict(float)

        for i in range(self.capacity): 
            dummy_lpn = "dummy"+str(i)
            if i <= int(preconditioning*self.capacity): 
                self.cache._write(dummy_lpn, state="A") # self["dummy"+str(i)] = "A"
                self.assigned._write(dummy_lpn) # self["dummy"+str(i)] = ""
            else:
                self.cache._write(dummy_lpn, state="A") # self["dummy"+str(i)] = "C"
                self.assigned._write(dummy_lpn) # self["dummy"+str(i)] = ""
            # TODO: change 0 to a random ppn in the address space and keep them striped across different channels.
            self.dirty_page_mapping[dummy_lpn] = 0

    def method2cache(self, method):
        if method == "LRU":
            return LRUCache()
        
    def read(self, lpn): # rw_cache.read(ext.lpn_start)
        writeback, wb_ppn = False, None
        if lpn not in self.cache:
            self.counter["read misses"] += 1
            hit = False
        else:
            self.counter["read hits"] += 1
            state = self.cache._read(lpn)
            if state == "CL":
                self.unassigned_cold._read(lpn)
            if state == "A":
                self.assigned._read(lpn)
            hit = True

        if writeback:
            self.counter["writebacks"] += 1
        
        return hit, writeback, wb_ppn
    

    def read_evict_cold(self, lpn):
        if lpn in self.evict_cold:
            hit = True
        else:
            hit = False
        return hit
    

    def read_and_delete(self, lpn):
        wb_ppn = None
        if lpn not in self.cache: 
            return
        else: 
            state = self.cache._read(lpn)
            if state == "A":
                self.assigned.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_cold._write(lpn)
            if state == "CL":
                return
        return wb_ppn
    
    def delete(self, lpn):
        if lpn in self.cache:
            state = self.cache._read(lpn)
            if state == "A":
                return
            if state == "CL":
                self.unassigned_cold.evict_lpn(lpn)
                self.cache.evict_lpn(lpn)
                self.evict_cold._write(lpn)
                self.inplace_update += 1

    
    def traversal_unassigned(self):
        unassigned_lpn_num = len(self.unassigned_cold) + len(self.assigned)
        inplace_update_num = self.inplace_update
        return unassigned_lpn_num, inplace_update_num

    def write(self, lpn, hotness):
        hit, writeback, wb_ppn = False, False, None
        if lpn not in self.cache:
            self.counter["write misses"] += 1
            self.cache._write(lpn, state="CL") 
            self.unassigned_cold._write(lpn)
            writeback, wb_ppn = self.evict(state_preference="A")
        else: # in cache
            hit = True
            self.counter["write hits"] += 1
            state = self.cache._read(lpn)
            if hotness == 'cold':
                if state == "CL":
                    self.unassigned_cold._read(lpn)
                    self.inplace_update += 1
                if state == "A":
                    self.assigned._read(lpn)
                    self.inplace_update += 1

        if writeback:
            self.counter["writebacks"] += 1
        
        return hit, writeback, wb_ppn
    
    def evict(self, state_preference): # cold
        writeback, wb_ppn = False, None
        if self.should_evict():
            lpn, state = self.cache.peak()
            if state == "A": 
                self.cache.evict_one()
                lpn, state = self.assigned.evict_lpn(lpn) 
                self.evict_cold._write(lpn)
                writeback, wb_ppn = True, self.dirty_page_mapping[lpn]
                del self.dirty_page_mapping[lpn]
            ## we still can hit U here if the size of unassigned is smaller than flush_size
            elif state == "CL": # state_preference="C"      
                if state_preference == "A" and len(self.assigned) != 0:
                    lpn, _ = self.assigned.evict_one()
                    _, state = self.cache.evict_lpn(lpn)
                    self.evict_cold._write(lpn)
                    assert(state == state_preference)
                    writeback, wb_ppn = True, self.dirty_page_mapping[lpn]
                    del self.dirty_page_mapping[lpn]
        
        if self.capacity + len(self.evict_cold) >= self.overall_data_cache_size:
            while self.capacity + len(self.evict_cold) >= self.overall_data_cache_size:
                evict_cold_lpn, _ = self.evict_cold.evict_one()
        
        return writeback, wb_ppn

    
    def update_assigned(self, mappings):
        for lpn, ppn in mappings.items():
            assert(lpn in self.assigned)
            self.dirty_page_mapping[lpn] = ppn 

    def select_unassigned(self, hotness):
        frame_length = 256
        split_results = defaultdict(list)
        candidates = []
        if hotness == 'cold':
            for i, (lpn, _) in enumerate(self.unassigned_cold.items()):
                split_results[lpn // frame_length].append((lpn, _))
                candidates.append(lpn)
                if i >= self.flush_size_cold - 1:
                    break


        per_frame_count = dict(sorted([(k, len(v)) for k,v in split_results.items()], key=lambda x:x[1], reverse=True))
        cumsum = np.cumsum(per_frame_count.values())

        ## select index as the last logical group to satisfy the flush size requirement
        index = np.argwhere(cumsum<=self.flush_size_cold*self.filter_ratio)[-1][0]
        #log_msg("index %d" % index)

        lpa_to_flush = sorted(filter(lambda x: x // frame_length in per_frame_count.keys()[:index+1], candidates)) 

        return lpa_to_flush


    def flush_unassigned(self, hotness):
        if hotness == 'cold':
            lpn_to_flush = self.select_unassigned(hotness = 'cold')
            for lpn in lpn_to_flush:
                self.assigned[lpn] = "" 
                del self.unassigned_cold[lpn]
                self.cache[lpn] = "A" # assigned 
                
        return lpn_to_flush

    def should_assign_page_cold(self):
        return (self.cache.peak()[1] == "CL") and len(self.cache) >= self.capacity and len(self.unassigned_cold) >= self.flush_size_cold 

    def should_evict(self):
        return len(self.cache) > self.capacity
    
    def remaining_size(self):
        return self.capacity - len(self.cache)  # cache_size / page_size

    def change_capacity(self, add_or_reduce, size):
        if add_or_reduce == True:
            self.capacity += size / self.slot_size
        else:
            self.capacity -= size / self.slot_size