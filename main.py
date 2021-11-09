#Add Patient
from pde.Patient import Patient 
from test_gui import App
import tkinter
import pickle
from pde.param_search import param_search
from config_loader import pde_config, data_config
def main():
    with open("predicted.pickle","rb") as f:
        data = pickle.load(f)
    p_list = list(data.keys())
    p_2 = []
    for p in p_list:
        p_2.append(Patient(p))
    params = pde_config.param
    start_step_c, start_step_t = params["start_step_c"], params["start_step_t"]
    range_c, range_t = eval(params["range_c"]), eval(params["range_t"])
    num_imgs = params["num_imgs"]
    for p in p_2:
        param_search(p.patient_info["id"],p.patient_info["wound_img"], start_step_c=start_step_c, range_c = range_c,
                    start_step_t = start_step_t, range_t = range_t, days = p.patient_info["days"],
                    num_imgs= num_imgs, reset=True) 

if __name__ == "__main__":
    main()

