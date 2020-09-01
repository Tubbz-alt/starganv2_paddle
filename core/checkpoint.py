"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import paddorch as porch


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        porch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        if not os.path.exists(fname):
            print(fname + ' does not exist!')
            return
        print('Loading checkpoint from %s...' % fname)
        if porch.cuda.is_available():
            module_dict = porch.load(fname)
        else:
            module_dict = porch.load(fname, map_location=porch.device('cpu'))
        for name, module in self.module_dict.items():
            if name in module_dict:
                print(name,"loaded")
                module.load_state_dict(module_dict[name])
