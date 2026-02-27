# Schistosomiasis Plugin for InVEST

In collaboration with the De Leo Lab at Stanford University we developed a tool
to incorporate schistosomiasis risk into dam and water infrastructure
development decisions. This work is based on the [Walz et al. 2015](https://doi.org/10.1371/journal.pntd.0004217) paper
with additional contributions by Andy Chamberlain and Giulio De Leo
from the De Leo Lab, and Lisa Mandle and Doug Denu from the Natural Capital Project.

## About Schistosomiasis

## About InVEST Plugins

## The model

Add info here about the schistosomiasis model, inputs, calculating risk.
Essentially could be the user guide section.

**Data*
- Spatial inputs must have the same projection and be projected with units of meters.

## Developer Guide

### Running in InVEST

Currently, plugins for InVEST is still under active development. A fork of
InVEST has been created to specifically support running the schisto plugin ([schisto-invest](https://github.com/natcap/schistosomiasis-invest)).
When users download InVEST to run the schistosomiasis plugin, they will be
using this custom version of InVEST. In the near future we hope that this
won’t be necessary and that users can use regular InVEST, but the following
custom functionality makes it necessary at the moment:
- Plugins. Schistosomiasis has been a great use case to help us develop the
idea of plugins, but we’re not quite ready to officially release the
functionality at scale.
- UI components for defining functions. We developed some bespoke UI
features to allow a better experience to define the shape of functions.
- Visualizing results with Jupyter Notebooks. We’re looking to bring this
type of functionality to InVEST generally, but are still early days in
designing and fleshing out use cases. Specifically, the version of InVEST
being used is found under the `plugin/schisto` branch of the repo.


**Download schist-invest**: [a link to download invest]

**Install plugin into schisto-invest**
To install the custom version of InVEST for schistosomiasis and the plugin
follow these steps.
1. Download and extract the executable from:
    a. [Windows installer](https://drive.google.com/file/d/1MIgofYmqJ6veqsUbBMfBIGdwJeOdgCFK/view?usp=sharing)
    b. [Mac installer](https://drive.google.com/file/d/1DPNuL3v9p4z8mRCdenX9kjyYWq6ONKCt/view?usp=sharing)
2. Run the executable. Currently you’ll get a notification that this
software isn’t trusted and will have to select “more info” to continue with
installing. This is because we aren’t currently code signing this with our
certificate. We can look into that as a possibility, if needed.
    a. Install for user only and NOT all users. There’s currently a permission
    limitation if installing system wide, which we’re looking to fix in the
    future.
3. Finish and launch the InVEST Workbench.
4. On first launch, a download modal will appear, you can cancel out of that.
5. In the upper right select “Manage plugins”.
6. Under “Add a plugin”, enter https://github.com/natcap/schistosomiasis.git.
Click “Add”.
7. This step will take several minutes. It is setting up an isolated python
environment, downloading, and installing the schistosomiasis model.
This is a one time step and won’t need to install the model again, unless
you make updates!
8. The schistosomiasis plugin should be installed and listed in the model list!

To run the schistosomiasis model, select the model from the list, which will
load the user interface for the model.

### The companion Jupyter Notebook

### Modifying the model

The core of the model lives in `src/natcap/invest/schistosomiasis.py`. 
