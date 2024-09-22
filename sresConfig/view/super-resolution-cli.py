"""
Super Resolution Application Programming Interface (API) for Weather/Climate Data Framework.
"""
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys

# Overriding classes in super-resolution-climate project
import veto.gpu
import veto.workflow
sys.modules["sres.base.gpu"] = veto.gpu
sys.modules["sres.controller.workflow"] = veto.workflow

from typing import Any, Dict, List
import time  # tracking time

from sresConfig.model.parms import parms
from sresConfig.controller.actions import ActionController

def main():
    """
    Main routine for Super Resolution Application Programming Interface (API).
    """
    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time
    print('Command line executed:    {'+str(sys.argv)+'}')

    # Initialize context
    contextClazz = parms(1)
    context = contextClazz.getDict()
    
    try:
        # Derive configuration names from CLI
        _task = f"{context[parms.SRES_VAR]}-tiles-{context[parms.SRES_TILESIZE]}"
        _dataset = f"swot_{context[parms.SRES_REGION]}"

        cname = context[parms.SRES_PIPELINE] #"sres"
        model =  context[parms.SRES_MODEL] #'swot' 
        models: List[str] = [ str(context[parms.SRES_MODEL]) ]
        ccustom: Dict[str,Any] = {}
        configuration = dict(
            task = _task,
            dataset = _dataset,
            pipeline = cname,
            platform = context[parms.SRES_PLATFORM],
         )

        # Process specified action
        if str(context[parms.SRES_ACTION]).endswith('train'):
            refresh =  False
            controller = ActionController( cname, configuration, epochs=context[parms.SRES_EPOCHS], 
                                          refresh_state=refresh, interp_loss=True )
            controller.train( models, **ccustom )
        elif str(context[parms.SRES_ACTION]).endswith('infer'):
            controller = ActionController( cname, configuration, structure=context[parms.SRES_STRUCTURE], 
                                          interp_loss=True )
            model = models[0]
            controller.infer( model, [ 0, int(context[parms.SRES_TIMESTEPS]) ], **ccustom )
        else:
            print("Invalid action = " + str(context[parms.SRES_ACTION]))
   
    except BaseException as err:
            print('\nWorkflow processing failed - Error details: ', err)

    elapsed_time = (time.time() - start_time) / 60.0
    print("\n" + str("{0:0.2f}".format(elapsed_time)) + " Total Elapsed Minutes for :" + str(context))


if __name__ == "__main__":
     main()
