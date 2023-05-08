from rbibm.scripts.rbibm_script import rbibm
from rbibm.scripts.collect_sweeps import collect_sweep_results
from rbibm.utils.utils_data import query_main

from datetime import date, datetime
import sys

ascii_logo = """          
                      
  _____  ____ _____ ____  __  __ 
 |  __ \|  _ \_   _|  _ \|  \/  |       --..,_                     _,.--.       
 | |__) | |_) || | | |_) | \  / |          `'.'.                .'`__ o  `;__.  
 |  _  /|  _ < | | |  _ <| |\/| |             '.'.            .'.'`  '---'`  ` 
 | | \ \| |_) || |_| |_) | |  | |              '.`'--....--'`.'            
 |_|  \_\____/_____|____/|_|  |_|                `'--....--'`         

 
                                           
"""

def main():
  """Main script to run"""
  print(ascii_logo)
  today = date.today()
  now = datetime.now()

  datum = today.strftime("%Y-%m-%d")
  time = now.strftime("%H-%M-%S")
  args = sys.argv
  
  print(args)
  # This is the hydra script
  rbibm()
  
  # Post hoc data collection + modifications...
  collect_sweep_results(datum, time, *args)