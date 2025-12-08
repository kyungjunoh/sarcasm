from torch.utils.data import DataLoader
from .dataset import VideoDataset


def get_train_dataloader(args):
    dataset = VideoDataset(args.datadir, subset='train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    return train_loader

def get_val_dataloader(args):
    dataset = VideoDataset(args.datadir, subset='val')
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def get_test_dataloader(args):
    dataset = VideoDataset(args.datadir, subset='test')
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return test_loader