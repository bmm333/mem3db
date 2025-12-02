## Dev journey: 14/11/2025 Decided to go with GPU VRAM as an L4 cache
This is going to take a long long time :D and lots of Coffeeee
What ive built till now is too cpu centric, will bring it to life just to benchmark it. Once that is done whole architecture needs to flip into SoA mindset.

-New Plan:
    -keys,values,hashes,flags - al split into separate flat buffers in VRam
    -fixed size keys/values for now but variabile later once i figure out a sane arena strategy.
    -everything in batches no small ops.
    -Cpu will act as the VRAM DataResident Manager
 Tested it out today, it was way slower than i thought , and its okay , was doing only 0.02m operations/sec , faster than std::unordered_map but nowhere near a inmemdb :D

 --Device_context.h
    |
    --> Simple gpu dynamic selection querying all avalible gpus with cudaGetDeviceCount and cudaMemGetInfo
    picking the device with most free vram (opt can maintain a gpu ranking table to switch devices in O(1) in case a larger batch wont fit). Singletone : Single deviceContext per process is fine for now , and thread safe access to steams and device info.

layout.cu -> assuming calls will succed , need's a wrapper for checking correctly likley : CUDA_CHECK(cudaMalloc....) with a fallback path(cpu mode) if gpu fails. 

right now now thread safe even though gpu kernels run in streams , host side allocator must be protected. 

Goals for now : 0 Fragmentations.

HashRouter: cpu side orchestrator. Managing staging bufferss to keep the data flowing smothly
hash_router.h: API used by server/client
hash_router.cpp: impl managing pinned memory and device staging buffers
using cudaMallocHost vs malloc where os can swap that part of memory to disk , so with malloc gpu cannot access it directly losing time to copy to a temp buffer first, instead using cudaMallocHost os promises to never move this memory so gpu can read at full PCIe speed 

FIRST RUN ON LOCAL: 
[INIT] Starting GPU Engine...
[Init] Allocating SoA Table (2097152 slots)...
[Init] Creating HashRouter...

--- Batch 1 / 5 ---
PUT: 2000000 items in 60.1884 ms (33 M ops/sec)
GET: Retrieved in 52.7344 ms (37 M ops/sec)
VALIDATION: 0 / 2000000 match correctly.

--- Batch 2 / 5 ---
PUT: 2000000 items in 40.2241 ms (49 M ops/sec)
GET: Retrieved in 49.1885 ms (40 M ops/sec)
VALIDATION: 1999972 / 2000000 match correctly.

--- Batch 3 / 5 ---
PUT: 2000000 items in 40.3857 ms (49 M ops/sec)
GET: Retrieved in 50.8257 ms (39 M ops/sec)
VALIDATION: 1999972 / 2000000 match correctly.

--- Batch 4 / 5 ---
PUT: 2000000 items in 46.0007 ms (43 M ops/sec)
GET: Retrieved in 50.1192 ms (39 M ops/sec)
VALIDATION: 1999972 / 2000000 match correctly.

--- Batch 5 / 5 ---
PUT: 2000000 items in 39.8104 ms (50 M ops/sec)
GET: Retrieved in 49.9526 ms (40 M ops/sec)
VALIDATION: 1999972 / 2000000 match correctly.

[Done] Total time: 1.47397s 
It took a few tweaks upgrades of cuda and downgrades so the flags need to be revied completly for compatibility. and also the sync at put and get of cuda, commenting out those lines at kernel , doubled the performance of the engine

notice: batch  1  should be considered as a warmup round instead when not using sync on put/get 
