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

