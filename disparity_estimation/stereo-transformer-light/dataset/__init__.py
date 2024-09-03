from torch.utils.data import DataLoader

from dataloader import get_dataset


def build_data_loader(args):
    
    
    dataset_train      = get_dataset(args.dataset, args.dataset_directory, "train", "STTR")
    dataset_validation = get_dataset(args.dataset, args.dataset_directory, 'validation', "STTR")
    dataset_test       = get_dataset(args.dataset, args.dataset_directory, "test", "STTR")


    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
    data_loader_validation = DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)

    return data_loader_train, data_loader_validation, data_loader_test