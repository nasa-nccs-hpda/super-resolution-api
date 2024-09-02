#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse  # system libraries
from datetime import datetime
from pathlib import Path
import csv

from enum import Enum
from pydantic import BaseModel,ValidationError

class Action(Enum):
    TRAIN = "train"
    INFER = "infer"

class Dataset(Enum):
    LLC = "LLC4320"
    MERRA2 = "merra2"
    SWOT = "swot"

class Model(Enum):
    DBPN = "dbpn"
    EDSR = "edsr"
    ESRT = "esrt"
    LAPSRN = "lapsrn"
    RCAN = "rcan"

class Pipeline(Enum):
    SRES = "sres"

class Platform(Enum):
    DESKTOP = "desktop"
    EXPLORE = "explore"

class SRESConfiguration(BaseModel):
    action: Action
    model: Model
    dataset: Dataset
    pipeline: Pipeline
    platform: Platform

# -----------------------------------------------------------------------------
# class context
#
# This class is a serializable context for orchestration.
# -----------------------------------------------------------------------------
class parms(object):
    # Custom name for current run
    BATCH_NAME = 'batch_name'
    CAT_ID = 'catid'

    # Super Resolution parameters
    SRES_BATCH = 'sres_batch'
    SRES_DEVICE = 'sres_device'
    SRES_DATASET = 'sres_dataset'
    SRES_MODEL = 'sres_model'
    SRES_PIPELINE = 'sres_pipeline'
    SRES_PLATFORM = 'sres_platform'
    SRES_TASK = 'sres_task'
    SRES_VAR = 'sres_var'
    SRES_ACTION = 'sres_action'
    SRES_VALIDATE_FLAG = 'sres_validate_flag'
    DIR_CONFIG = 'sres_config_dir'

    # Global instance variables
    context_dict = {}
    plotLib = None
    debug_level = 0
    writer = None

    # -------------------------------------------------------------------------
    # trace()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def trace(self, value):
        if (self._debug_level > 0):
            print(value)

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, debug_level):
        self._debug_level = debug_level
        args = self._getParser()
        # Initialize serializable context for orchestration
        try:
            self.context_dict[parms.SRES_VALIDATE_FLAG] = str(args.validatebool)
            config_location = str(args.sres_config_dir)
            if (os.path.isdir(config_location)):
                self.context_dict[parms.DIR_CONFIG] = config_location
            else:
                raise FileNotFoundError("Config directory not found: {}".format(config_location))
                
            self.context_dict[parms.SRES_BATCH] = str(args.sres_batch)
            self.context_dict[parms.SRES_DEVICE] = str(args.sres_device)
            self.context_dict[parms.SRES_VAR] = str(args.sres_var)            
            self.context_dict[parms.SRES_TASK] = str(args.sres_task)            
            self.context_dict[parms.SRES_DATASET] = str(args.sres_dataset)            
            self.context_dict[parms.SRES_MODEL] = str(args.sres_model)            
            self.context_dict[parms.SRES_PIPELINE] = str(args.sres_pipeline)            
            self.context_dict[parms.SRES_ACTION] = str(args.sres_action)            
            self.context_dict[parms.SRES_PLATFORM] = str(args.sres_platform)            

            if eval(self.context_dict[parms.SRES_VALIDATE_FLAG]):
                try:
                    # class SRESConfiguration(BaseModel):
                    #     action: Action
                    #     model: Model
                    #     dataset: Dataset
                    #     pipeline: Pipeline
                    #     platform: Platform
                    srescfg = SRESConfiguration( 
                        action=self.context_dict[parms.SRES_ACTION],
                        model=self.context_dict[parms.SRES_MODEL],
                        dataset=self.context_dict[parms.SRES_DATASET],
                        pipeline=self.context_dict[parms.SRES_PIPELINE],
                        platform=self.context_dict[parms.SRES_PLATFORM]
                    )
                    print(srescfg)
                except ValidationError as e:
                        print(e)
                        sys.exit(1)

        except BaseException as err:
            print('Check arguments: ', err)
            sys.exit(1)

        # Initialize instance variables
#        self.debug_level = int(self.context_dict[parms.DEBUG_LEVEL])

        # Echo input parameter values
    # SRES_VAR = 'sres_var'
    # SRES_TASK = 'sres_task'
    # SRES_DATASET = 'sres_dataset'
    # SRES_PIPELINE = 'sres_pipeline'
    # SRES_PLATFORM = 'sres_platform'

        self.trace(f'Initializing Super Resolution invocation with the following parameters:')
        if (self.context_dict[parms.SRES_BATCH] != None): self.trace(f'Batch:    {self.context_dict[parms.SRES_BATCH]}')
        if (self.context_dict[parms.SRES_DEVICE] != None): self.trace(f'Device:    {self.context_dict[parms.SRES_DEVICE]}')
        self.trace(f'Action:    {self.context_dict[parms.SRES_ACTION]}')
        self.trace(f'Dataset:    {self.context_dict[parms.SRES_DATASET]}')
        self.trace(f'Model:    {self.context_dict[parms.SRES_MODEL]}')
        self.trace(f'Pipeline:    {self.context_dict[parms.SRES_PIPELINE]}')
        self.trace(f'Platform:    {self.context_dict[parms.SRES_PLATFORM]}')
        self.trace(f'Task:    {self.context_dict[parms.SRES_TASK]}')
        self.trace(f'Var:    {self.context_dict[parms.SRES_VAR]}')
        self.trace(f'Validation:    {self.context_dict[parms.SRES_VALIDATE_FLAG]}')

        return

    # -------------------------------------------------------------------------
    # getParser()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _getParser(self):
        """
        :return: argparser object with CLI commands.
        """
        parser = argparse.ArgumentParser()

        # Super Resolution parameters
        # var = "sst"
        # task = "cape_basin",
        # dataset = "LLC4320",
        # pipeline = "sres",
        # platform = "explore_gt"
        
        parser.add_argument(
            "--batch", "--batch", type=str, required=False, dest='sres_batch',
            default=None, help="Specify batch name for run."
        )
        parser.add_argument(
            "--device", "--device", type=str, required=False, dest='sres_device',
            default=None, help="Override default device name for run."
        )
        parser.add_argument(
            "-config_dir", "--config_dir", type=str, required=True, dest='sres_config_dir',
            default=None, help="Specify directory path containing CONFIG files."
        )
        parser.add_argument(
            "-dataset", "--dataset", type=str, required=True, dest='sres_dataset',
            default="LLC4320", help="Specify super-resolution dataset to process."
        )
        parser.add_argument(
            "-model", "--model", type=str, required=True, dest='sres_model',
            default="rcan", help="Specify super-resolution model to process."
        )
        parser.add_argument(
            "-pipeline", "--pipeline", type=str, required=True, dest='sres_pipeline',
            default="sres", help="Specify super-resolution pipeline to process."
        )
        parser.add_argument(
            "-platform", "--platform", type=str, required=True, dest='sres_platform',
            default="explore_gt", help="Specify super-resolution platform to process."
        )
        parser.add_argument(
            "-task", "--task", type=str, required=True, dest='sres_task',
            default="cape_basin", help="Specify super-resolution task to process."
        )
        parser.add_argument(
            "-var", "--var", type=str, required=True, dest='sres_var',
            default="sst", help="Specify super-resolution variable to process."
        )
        parser.add_argument(
            "-action", "--action", type=str, required=True, dest='sres_action',
            help="Specify super-resolution action to process (e.g., train, infer)."
        )
        parser.add_argument(
            "--debug", "--debug_level", type=int, required=False, dest='debug_level',
            default=0, help="Specify debug level [0,1,2,3]"
        )
        parser.add_argument(
            "--validate", "--validate", required=False, dest='validatebool',
            action='store_true', help="Validate input parameters at startup."
        )
        parser.add_argument(
            "--log", "--log", required=False, dest='logbool',
            action='store_true', help="Set logging."
        )
        return parser.parse_args()

    # -------------------------------------------------------------------------
    # getDict()
    #
    # Get context dictionary
    # -------------------------------------------------------------------------
    def getDict(self):
        return self.context_dict


    # # -------------------------------------------------------------------------
    # # getDebugLevel()
    # #
    # # Get debug_level
    # # -------------------------------------------------------------------------
    # def getDebugLevel(self):
    #     return self.debug_level

    # -------------------------------------------------------------------------
    # getFileNames()
    #
    # Get input file names
    # -------------------------------------------------------------------------
    def getFileNames(self, prefix, context):
        """
        :param prefix: core TOA file name (must match core target and cloudmask file name)
        :param context: input context object dictionary
        :return: updated context
        """
        context[parms.FN_PREFIX] = str((prefix[1]).split("-toa.tif", 1)[0])
        last_index = context[parms.FN_PREFIX].rindex('_')
        context[parms.CAT_ID] =  context[parms.FN_PREFIX] [last_index+1:]

        # Provide the fully-qualified file name (if provided).  Otherwise assume, list of files
        if os.path.isfile(Path(context[parms.DIR_TOA])):
            context[parms.FN_TOA] = context[parms.DIR_TOA]
        else:
            context[parms.FN_TOA] = os.path.join(context[parms.DIR_TOA] + '/' +
                                                   context[parms.FN_PREFIX] + context[parms.FN_TOA_SUFFIX])

        if os.path.isfile(Path(context[parms.DIR_TARGET])):
            context[parms.FN_TARGET] = context[parms.DIR_TARGET]
        else:
            context[parms.FN_TARGET] = os.path.join(context[parms.DIR_TARGET] + '/' +
                                                      context[parms.FN_PREFIX] + context[parms.FN_TARGET_SUFFIX])

        if os.path.isfile(Path(context[parms.DIR_CLOUDMASK])):
            context[parms.FN_CLOUDMASK] = context[parms.DIR_CLOUDMASK]
        else:
            context[parms.FN_CLOUDMASK] = os.path.join(context[parms.DIR_CLOUDMASK] + '/' +
                                                         context[parms.FN_PREFIX] + context[
                                                             parms.FN_CLOUDMASK_SUFFIX])

        # Name artifacts according to TOA prefix
        context[parms.FN_TOA_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                         context[parms.FN_PREFIX] + self.FN_TOA_DOWNSCALE_SUFFIX)
        context[parms.FN_TARGET_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                            context[
                                                                parms.FN_PREFIX] + self.FN_TARGET_DOWNSCALE_SUFFIX)
        context[parms.FN_CLOUDMASK_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                               context[
                                                                   parms.FN_PREFIX] + self.FN_CLOUDMASK_DOWNSCALE_SUFFIX)

                                                    
        if (eval(self.context_dict[parms.NONCOG_FLAG])):
            context[parms.FN_COG] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                               context[parms.FN_PREFIX] + self.FN_SRLITE_NONCOG_SUFFIX)
        else:
            context[parms.FN_COG] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                               context[parms.FN_PREFIX] + self.FN_SRLITE_SUFFIX)


        if not (os.path.exists(context[parms.FN_TOA])):
            raise FileNotFoundError("TOA File not found: {}".format(context[parms.FN_TOA]))
        if not (os.path.exists(context[parms.FN_TARGET])):
            self.plot_lib.trace("Processing: " + context[parms.FN_TOA])
            raise FileNotFoundError("TARGET File not found: {}".format(context[parms.FN_TARGET]))
        if (eval(self.context_dict[parms.CLOUD_MASK_FLAG])):
            if not (os.path.exists(context[parms.FN_CLOUDMASK])):
                self.plot_lib.trace("Processing: " + context[parms.FN_TOA])
                raise FileNotFoundError("Cloudmask File not found: {}".format(context[parms.FN_CLOUDMASK]))

        return context

    # -------------------------------------------------------------------------
    # _create_logfile()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _create_logfile(self, model, logdir='results'):
        """
        :param args: argparser object
        :param logdir: log directory to store log file
        :return: logfile instance, stdour and stderr being logged to file
        """
        logfile = os.path.join(logdir, '{}_log_{}_model.txt'.format(
            datetime.now().strftime("%Y%m%d-%H%M%S"), model))
        print('See ', logfile)
        so = se = open(logfile, 'w')  # open our log file
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
        os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
        os.dup2(se.fileno(), sys.stderr.fileno())
        return logfile
