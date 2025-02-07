STORM
======
A python software for Storm Tracking On-orbit using RFS with Complementary MOS (STORM) developed by the Laboratory for Autonomy, GNC, and Estimation Research (LAGER) at the University of Alabama (UA)

## Dev environment
To facilitate development of STORM which requires concurrent development in CARBS, ```git submodules``` is used. CARBS submodule tracks the ```ggiw-dev``` [branch](https://github.com/drjdlarson/carbs/tree/ggiw-dev).

Clone the repository locally on your computer, using

```
git clone --recurse-submodules git@github.com:TL-4319/storm.git
```

For windows it is recommended to clone it within your linux subsystem directory (e.g. a sub-directory of your linux home folder) to improve performance within the container (the linux directories on Windows can be accessed through the file browser by typing ```\\wsl$``` in the address bar and clicking on your distro).

It is highly recommended to use VS Code with the dev containers extension. 