import itertools
from typing import List, Dict
from argparse import ArgumentError

fgsm_arguments = ["attack", "attack_norm", "attack_epsilon", "attack_alpha", "attack_targeted", "attack_target", "attack_loss"]
bim_pgd_cospgd_arguments = ["attack", "attack_norm", "attack_epsilon", "attack_alpha", "attack_targeted", "attack_target", "attack_loss", "attack_iterations"]
apgd_arguments = ["attack", "attack_norm", "attack_epsilon", "attack_targeted", "attack_target", "attack_loss", "attack_iterations"]
pcfa_arguments = ["attack", "attack_targeted", "attack_target", "attack_loss", "pcfa_delta_bound", "pcfa_steps", "pcfa_boxconstraint"]
targeted_arguments = ["attack_target"]

class AttackArgumentParser:
    def __init__(self, args) -> None:
        self.attack_args = {}
        self.argument_lists = []
        self.number_of_args = 0
        self.index = 0
        for arg in vars(args): 
            if arg.startswith("attack") or arg.startswith("pcfa"):
                self.attack_args[arg] = list(set(self.to_list(getattr(args, arg))))
        self.number_of_args = len(self.attack_args.keys())
        self.args_list_to_arg_sets()
        self.filter_arguments()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.argument_lists):
            raise StopIteration
        else:
            argument_dic = self.argument_lists[self.index]
            self.index = self.index + 1
            return argument_dic

    def args_list_to_arg_sets(self):
        args_list = []
        for arg_list in self.attack_args.values():
            args_list.append(arg_list)
        args_list_flat = self.flatten_and_combine(args_list)
        
        args_list = [args_list_flat[i:i + self.number_of_args] for i in range(0, len(args_list_flat), self.number_of_args)]
        for entry in args_list:
            arguments = {}
            counter = 0
            for arg_name in self.attack_args.keys():
                arguments[arg_name] = entry[counter]
                counter = counter + 1
            self.argument_lists.append(arguments)


    def flatten_and_combine(self, lists):
        if len(lists) == 1:
            return lists[0]
        else:
            # Generate all combinations from sublists first
            rest_combined = list(itertools.product(*lists[1:]))
            result = []
            for item in lists[0]:
                for combo in rest_combined:
                    result.extend([item] + list(combo))
            return result

    def to_list(self, var):
            if not isinstance(var, list):
                return [var]
            return var
    
    def filter_arguments(self):
        for i in range(0, len(self.argument_lists)):
            entry = self.argument_lists[i]
            attack = entry["attack"]
            targeted = entry["attack_targeted"]
            to_remove = dict(entry)
            
            match attack: # Commit adversarial attack
                case "fgsm":
                    for arg_name in entry.keys():
                        if arg_name not in fgsm_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                case "pgd" | "cospgd" | "bim":
                    for arg_name in entry.keys():
                        if arg_name not in bim_pgd_cospgd_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                case "apgd":
                    for arg_name in entry.keys():
                        if arg_name not in apgd_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                case "pcfa":
                    for arg_name in entry.keys():
                        if arg_name not in pcfa_arguments:
                            del to_remove[arg_name]
            self.argument_lists[i] = to_remove

        indexes_to_remove = set()
        for i in range(0, len(self.argument_lists)):
            if i in indexes_to_remove:
                continue
            for j in range(i + 1, len(self.argument_lists)):
                if self.argument_lists[i] == self.argument_lists[j]:
                    indexes_to_remove.add(j)

        new_argument_list = []
        for i in range(0, len(self.argument_lists)):
            if i not in indexes_to_remove:
                new_argument_list.append(self.argument_lists[i])
        self.argument_lists = new_argument_list
        self.filter_pcfa_untargeted()
            
    def filter_pcfa_untargeted(self):
        pcfa_targeted_flag = False
        pcfa_untargeted_indices = []
        for i in range(0, len(self.argument_lists)):
            if not self.argument_lists[i]["attack"] == "pcfa":
                continue
            else:
                if not self.argument_lists[i]["attack_targeted"]:
                    pcfa_untargeted_indices.append(i)
                else:
                    pcfa_targeted_flag = pcfa_targeted_flag or self.argument_lists[i]["attack_targeted"]     
        if pcfa_targeted_flag and len(pcfa_untargeted_indices) > 0:
            new_argument_list = []
            for i in range(0, len(self.argument_lists)):
                if i not in pcfa_untargeted_indices:
                    new_argument_list.append(self.argument_lists[i])
            self.argument_lists = new_argument_list

def attack_targeted_string(s):
    if s.upper() in {'TRUE', '1'}:
        return True
    elif s.upper() in {'FALSE', '0'}:
        return False
    else:
        raise ValueError('Not a valid boolean string')
    

                

    






    

