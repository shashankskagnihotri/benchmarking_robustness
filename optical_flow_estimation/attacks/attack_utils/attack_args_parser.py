import itertools
from typing import List
from argparse import ArgumentError


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

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.argument_lists):
            raise StopIteration
        else:
            argument_list = self.argument_lists[self.index]
            arguments = {}
            counter = 0
            for arg_name in self.attack_args.keys():
                arguments[arg_name] = argument_list[counter]
                counter = counter + 1
            self.index = self.index + 1
            return arguments

    def args_list_to_arg_sets(self):
        args_list = []
        for arg_list in self.attack_args.values():
            args_list.append(arg_list)
        args_list_flat = self.flatten_and_combine(args_list)
        
        args_list = [args_list_flat[i:i + self.number_of_args] for i in range(0, len(args_list_flat), self.number_of_args)]
        self.argument_lists = args_list

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





    

