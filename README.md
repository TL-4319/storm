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

## Workflow
After a dev container is spun up, the shell is logged in as non-root user ```vscode``` so that any file created is not root-acess limited in the host machine. NOTE: vscode still has root proviledge in the containter.

The terminal by default will not have a shell environment i.e. shows up as just ```$```. Enter ```bash``` to begin a bash environment for nice features.

### Development in CARBS
After implementing new feature in CARBS submodule and ready for test withint the container, build CARBS locally. NOTE: Don't forget "./" prefixing carbs to install the correct version.

```
pip3 install -e ./carbs
```

CARBS submodule is tracked via its own git and needs to be commit and push from within ```storm/carbs```. Submodule git commit then follows accordingly.

### Developing STORM application
If a code requires import from CARBS, make sure to build the latest local CARBS package.