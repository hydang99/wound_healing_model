from .adi_solver import ADI_solver
from config_loader import pde_config
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_logger
logger = get_logger("PARAM_SEARCH")
global num_iter 
num_iter = 0
def param_search(id,wound_imgs, start_step_c, start_step_t, range_c, range_t, days, num_imgs, reset = False):
    #Initialize variables
    """
    param_dict = save loss of all parameters combinations
    start_step_c = 0.1
    start_step_t = 0.1
    range_c = (should be 0 - 2)
    range_t = (always 0 - 1)
    """
    #Initialize the dict to have the form dict [constant][threshold]
    logger.info("PARAM Search")
    def param_search_t(img, constant, param_dict, t_range_info, best_thresh, init = True, heal = True):
        if constant not in param_dict.keys():
            param_dict[constant] = dict() 
        def heal_thresh(best_thresh):
            min_value = float("inf")
            cont = True
            heal = True
            while(heal == True and cont == True):
                for thresh in (np.arange(best_thresh, t_range_info[1], t_range_info[2])):
                    thresh = round(thresh,4)             
                    if(cont == True):
                        sum_match = 0
                        _, imgs = ADI_solver(id,begin_img, dx, dy, dt, 
                                                 max(days), constant,
                                                day = str(days), thresh = thresh, reset = reset)
                        for i in range(num_imgs): 
                            img = imgs[days[i] - 2]
                            compare_img = compare_set[i]
                            sum_match += abs(len(img[img==0]) - len(compare_img[compare_img ==0]))
                        param_dict[constant][thresh] = int(sum_match/num_imgs)
                        if sum_match < min_value:
                            min_value = sum_match
                            cont = True
                        else:
                            cont = False
                            break
                        img_last = imgs[len(imgs)-1]
                        img_first = imgs[0]
                        if len(img_last[img_last==0]) - len(img_first[img_first ==0]) < 0:
                            heal = True
                        else:
                            heal = False
            return min_value
        
    
        def un_heal_thresh(best_thresh):
            cont = True
            min_value = float("inf")
            heal = False
            while(heal == False and cont == True):
                for thresh in (np.arange(best_thresh, t_range_info[0], -t_range_info[2])):
                    thresh = round(thresh,4)
                    if(cont == True):
                        sum_match = 0
                        _, imgs = ADI_solver(id,begin_img, dx, dy, dt, 
                                                 max(days), constant,
                                                day = str(days), thresh = thresh,reset=reset)
                        for i in range(num_imgs): 
                            img = imgs[days[i] - 2]
                            compare_img = compare_set[i]
                            sum_match += abs(len(img[img==0]) - len(compare_img[compare_img ==0]))
                        param_dict[constant][thresh] = int(sum_match/num_imgs)
                        if sum_match < min_value:
                            min_value = sum_match
                            cont = True
                        else:
                            cont = False
                            break
                        img_last = imgs[len(imgs)-1]
                        img_first = imgs[0]
                        if len(img_last[img_last==0]) - len(img_first[img_first ==0]) < 0:
                            heal = True
                        else:
                            heal = False
            return min_value
                
        
        if init == True:
            min_heal = heal_thresh(t_range_info[0])
            min_unheal = un_heal_thresh(t_range_info[1])
            if(min_unheal < min_heal):
                heal = False
            best_t_init = min(param_dict[constant],key=param_dict[constant].get)
            final_result[str((constant, best_t_init))] = param_dict[constant][best_t_init]
            return heal, best_t_init
        
        else:
            if heal:
                min_heal = heal_thresh(best_thresh)
                best_t_init = min(param_dict[constant],key=param_dict[constant].get)
                final_result[str((constant, best_t_init))] = param_dict[constant][best_t_init]
                return min_heal, best_t_init
            else:
                min_unheal = un_heal_thresh(best_thresh)
                best_t_init = min(param_dict[constant],key=param_dict[constant].get)
                final_result[str((constant, best_t_init))] = param_dict[constant][best_t_init]
                return min_unheal, best_t_init

            
    def param_search_c(img, param_dict, c_range_info, best_t,heal, keep_t = False):
        min_value = float("inf")
        delay = 0
        old_thresh = best_t
        for constant in (np.arange(c_range_info[0], c_range_info[1], c_range_info[2])):
            constant = round(constant,5)
            if(delay < 3 and keep_t == False):
                best_match, new_best_thresh = param_search_t(img, constant, param_dict,
                                                             t_range_info = t_range, init = False, 
                                                             best_thresh = old_thresh, heal = heal)
                if not (new_best_thresh == old_thresh):
                    old_thresh = new_best_thresh
                    delay = 0
                else:
                    delay+=1
            else:
                if constant not in param_dict.keys():
                    param_dict[constant] = dict() 
                sum_match = 0
                _, imgs = ADI_solver(id,begin_img, dx, dy, dt, 
                         max(days), constant,
                        day = str(days), thresh = old_thresh,reset=reset)
                for i in range(num_imgs): 
                    img = imgs[days[i] - 2]
                    compare_img = compare_set[i]
                    sum_match += abs(len(img[img==0]) - len(compare_img[compare_img ==0]))
                    param_dict[constant][old_thresh] = int(sum_match/num_imgs)
                if(sum_match <= min_value):
                    min_value = sum_match
                    final_result[str((constant, old_thresh))] = int(sum_match/num_imgs)
                else:
                    break

                
            
        return final_result
    
    param_dict = dict()
    compare_set = list()
    
    days = days[:num_imgs]
    for index in range(len(days)):
        compare_set.append(wound_imgs["day_{}".format(days[index])])
    begin_img = wound_imgs['day_{}'.format(days[0])]
    logger.info("Search {}".format(days[0]))
    #print("Considering Image At Day {}".format(days[0]))
    plt.imshow(begin_img)
    
    #Some Params Definition
    dt = 1
    dx, dy = 1,1
    
    init_constant = start_step_c
    final_result = dict()
    t_range = (range_t[0], range_t[1], start_step_t)
    heal, best_t_init = param_search_t(begin_img, init_constant, param_dict, best_thresh = None,t_range_info = t_range)
    c_range_info = (range_c[0], range_c[1],start_step_c)
    param_search_c(begin_img, param_dict, c_range_info, best_t_init, heal = heal)
    min_constant = eval(min(final_result, key = final_result.get))
    t_range = (min_constant[1] - start_step_t, min_constant[1] + start_step_t, start_step_t)
    start_step_c/=8
    c_range_new = (min_constant[0] - 0.1, min_constant[1] + 0.1, start_step_c)
    param_search_c(begin_img, param_dict, c_range_new, min_constant[1], heal = heal, keep_t = True)
    return min(final_result,key=final_result.get), param_dict
def eval_param(wound_imgs, days, param):
    return None