
import  os
import  torch
import  utils
import  torch.nn    as      nn
import  pandas      as      pd
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm
import time
import torch.nn.functional as F
from skimage import color

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
    
class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def normal_accuracy(labels_pred,labels):    
    _, predicted = torch.max(labels_pred.data, 1)
    total = labels.size(0)
    # return (((predicted == labels).sum().item())/total)*100
    return torch.tensor([0.000000000000001])

def train_MSE(
    train_loader,
    model,
    sleep ,
    model_name,
    epochs,
    learning_rate,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
    
    val_loader,
    test_ealuate:bool,
    tets_loader,
    
    total_iters:int=20,
    ):

    model = model.to(device)

    # loss function
    criterion   = nn.MSELoss()
    # optimzier
    optimizer =  torch.optim.Adam(params=model.parameters(),
                                    betas= [0.9, 0.99],
                                    lr=learning_rate,
                                    # momentum=utils.opt_momentum,
                                    # weight_decay=utils.weight_decay
                                    )

    if load_saved_model:
        model, optimizer = load_model(
                                      ckpt_path=ckpt_path, model=model, optimizer=optimizer
                                        )

    lr_schedulerr =  lr_scheduler.LinearLR(optimizer,
                                           start_factor=utils.start_factor,
                                           end_factor=utils.end_factor,
                                           total_iters=total_iters)
    
    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_loss_till_current_batch",
            "avg_val_loss_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()        
        
        loop_train = tqdm(
                            enumerate(train_loader),#enumerate(train_loader,1),
                            total=len(train_loader),
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_dum=[]
        mode = "train"    
        for batch_idx, (lumination,aandb) in loop_train:
            images = lumination.to(device)
            aandb = aandb.to(device)
            
            labels_pred = model(images)
            
            loss = criterion(labels_pred, aandb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_avg_train.update(loss.item(), images.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_loss_till_current_batch":loss_avg_train.avg,
                 "avg_val_loss_till_current_batch":None},index=[0]
                 )

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                max_len=2,
                refresh=True,
            )
            time.sleep(sleep)
            
            
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            
            accuracy_dum=[]
            for batch_idx, (lumination,aandb) in loop_val:
                optimizer.zero_grad()
                images = lumination.to(device)
                aandb = aandb.to(device)
                
                labels_pred = model(images)
                
                loss = criterion(labels_pred, aandb)

                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg },index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    refresh=True,
                )
        if test_ealuate==True:
            mode = "test"
            with torch.no_grad():
                loop_val = tqdm(
                                enumerate(tets_loader, 1),
                                total=len(tets_loader),
                                desc="test",
                                position=0,
                                leave=True,
                                )
                accuracy_dum=[]
                for batch_idx, (lumination,aandb) in loop_val:
                    optimizer.zero_grad()
                    images = lumination.to(device)
                    aandb = aandb.to(device)
                    
                    labels_pred = model(images)
                    
                    loss = criterion(labels_pred, aandb)

            
                    loss_avg_val.update(loss.item(), images.size(0))
                    new_row = pd.DataFrame(
                        {"model_name": model_name,
                        "mode": mode,
                        "epoch": epoch,
                        "learning_rate":optimizer.param_groups[0]["lr"],
                        "batch_size": images.size(0),
                        "batch_index": batch_idx,
                        "loss_batch": loss.detach().item(),
                        "avg_train_loss_till_current_batch":None,
                        "avg_val_loss_till_current_batch":loss_avg_val.avg },index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"test - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        refresh=True,
                    )    
            
        lr_schedulerr.step()
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    torch.save(model.state_dict(), report_path+'/'+model_name+'.pt')
    return model, optimizer, report





def train(
    train_loader,
    model,
    sleep ,
    model_name,
    epochs,
    learning_rate,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
    
    tets_loader:torch.utils.data.DataLoader,
    
    total_iters:int=20,
    ):

    model = model.to(device)

    # loss function
    criterion_MsE   = nn.MSELoss()
    criterion_Cross = nn.CrossEntropyLoss()
    # optimzier
    optimizer =  torch.optim.Adam(params=model.parameters(),
                                    betas= [0.9, 0.99],
                                    lr=learning_rate,
                                    # momentum=utils.opt_momentum,
                                    weight_decay=utils.weight_decay)

    if load_saved_model:
        model, optimizer = load_model(
                                      ckpt_path=ckpt_path, model=model, optimizer=optimizer
                                        )

    lr_schedulerr =  lr_scheduler.LinearLR(optimizer,
                                           start_factor=utils.start_factor,
                                           end_factor=utils.end_factor,
                                           total_iters=total_iters)
    
    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_loss_till_current_batch",
            "avg_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()        
        
        loop_train = tqdm(
                            enumerate(train_loader),#enumerate(train_loader,1),
                            total=len(train_loader),
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_dum=[]
            
        for batch_idx, (lumination,aandb) in loop_train:
            images = lumination.to(device)
            aandb = aandb.to(device)
            # labels = labels.to(device)
            
            labels_pred = model(images)
            
            # Predicted = postprocess_tens(images,labels_pred)
            # Predicted = Predicted.to(device=device)
            
            loss = criterion(labels_pred, aandb)#/aandb.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1 = normal_accuracy(labels_pred, aandb)#accuracy(labels_pred, labels)
            accuracy_dum.append(acc1)
            acc1 = sum(accuracy_dum)/len(accuracy_dum)
            
            loss_avg_train.update(loss.item(), images.size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_loss_till_current_batch":loss_avg_train.avg,
                 "avg_acc_till_current_batch":acc1,},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                # accuracy_train="{:.4f}".format(acc1),
                max_len=2,
                refresh=True,
            )
            time.sleep(sleep)
            
            
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            acc1 = 0
            total = 0
            accuracy_dum=[]
            for batch_idx, (images, labels) in loop_val:
                optimizer.zero_grad()
                images = images.to(device).float()
                labels = labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                
                acc1 =normal_accuracy(labels_pred,labels)
                accuracy_dum.append(acc1)
                acc1 = sum(accuracy_dum)/len(accuracy_dum)

                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_acc_till_current_batch":acc1},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    accuracy_val="{:.4f}".format(acc1),
                    refresh=True,
                )
        if test_ealuate==True:
            mode = "test"
            with torch.no_grad():
                loop_val = tqdm(
                                enumerate(tets_loader, 1),
                                total=len(tets_loader),
                                desc="test",
                                position=0,
                                leave=True,
                                )
                accuracy_dum=[]
                for batch_idx, (images, labels) in loop_val:
                    optimizer.zero_grad()
                    images = images.to(device).float()
                    labels = labels.to(device)
                    labels_pred = model(images)
                    loss = criterion(labels_pred, labels)
                    
                    acc1 =normal_accuracy(labels_pred,labels)
                    accuracy_dum.append(acc1)
                    acc1 = sum(accuracy_dum)/len(accuracy_dum)
            
                    loss_avg_val.update(loss.item(), images.size(0))
                    new_row = pd.DataFrame(
                        {"model_name": model_name,
                        "mode": mode,
                        "image_type":"original",
                        "epoch": epoch,
                        "learning_rate":optimizer.param_groups[0]["lr"],
                        "batch_size": images.size(0),
                        "batch_index": batch_idx,
                        "loss_batch": loss.detach().item(),
                        "avg_train_loss_till_current_batch":None,
                        "avg_train_acc_till_current_batch":None,
                        "avg_val_loss_till_current_batch":loss_avg_val.avg,
                        "avg_val_acc_till_current_batch":acc1},index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"test - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        accuracy_val="{:.4f}".format(acc1),
                        refresh=True,
                    )    
            
        lr_schedulerr.step()
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    torch.save(model.state_dict(), report_path+'/'+model_name+'.pt')
    return model, optimizer, report