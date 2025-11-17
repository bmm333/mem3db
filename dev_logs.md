## Dev journey: 14/11/2025 Decided to go with GPU VRAM as an L4 cache
This is going to take a long long time :D and lots of Coffeeee
What ive built till now is too cpu centric, will bring it to life just to benchmark it. Once that is done whole architecture needs to flip into SoA mindset.

-New Plan:
    -keys,values,hashes,flags - al split into separate flat buffers in VRam
    -fixed size keys/values for now but variabile later once i figure out a sane arena strategy.
    -everything in batches no small ops.
    -Cpu will act as the VRAM DataResident Manager
 