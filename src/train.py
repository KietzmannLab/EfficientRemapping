import sys
import torch
from typing import Callable
import functions
from ModelState import ModelState
from Dataset import Dataset
import wandb


def test_epoch(ms: ModelState,
               dataset: Dataset,
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
               batch_size: int,
               sequence_length: int,
               mnist=False):
    tot_loss = 0
    tot_res = None 
    state = None
    if mnist:
        batches, fixations = dataset.create_list_batches(batch_size=batch_size, sequence_length=sequence_length, shuffle=True)
        num_batches = batches.shape[0]
        for i, batch in enumerate(batches):
            with torch.no_grad():
                loss, res, state = test_batch(ms, batch, fixations[i], loss_fn, state)
        
            tot_loss += loss

            if tot_res is None:
                tot_res = res
            else:
                tot_res += res
    else:
        loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
        num_batches = len(dataset) // batch_size + 1
        for batch, fixation in loader:
            with torch.no_grad():
                loss, res, state = test_batch(ms, batch, fixation, loss_fn, state)
        
            tot_loss += loss

            if tot_res is None:
                tot_res = res
            else:
                tot_res += res
  
    tot_loss /= num_batches
    tot_res /= num_batches
    print("Test loss:     {:.8f}".format(tot_loss))
    return tot_loss, tot_res

def test_batch(ms: ModelState,
               batch: torch.FloatTensor,
               fixations,
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
               state) -> float:
    loss, res, state = ms.run(batch, fixations, loss_fn, state)
    return loss.item(), res, state
    
def train_batch(ms: ModelState,
                batch: torch.FloatTensor,
                fixations,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                state,
                scaler=None) -> float:

    loss, res, state = ms.run(batch, fixations, loss_fn, state)
   
    ms.step(loss, scaler)
    ms.zero_grad()
    return loss.item(), res, state

def train_epoch(ms: ModelState,
                dataset: Dataset,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                batch_size: int,
                sequence_length: int,
                verbose = True,
                mnist=False,
                scaler=None) -> float:

    t = functions.Timer()
    tot_loss = 0.
    tot_res = None
    state = None
    if mnist:
        batches, fixations = dataset.create_list_batches(batch_size=batch_size, sequence_length=sequence_length, shuffle=True)

        num_batches = batches.shape[0]

        for i, batch in enumerate(batches):

            loss, res, state = train_batch(ms, batch, fixations[i], loss_fn, state)
            tot_loss += loss

            if tot_res is None:
                tot_res = res
            else:
                tot_res += res

            if verbose and (i+1) % int(num_batches/10) == 0:
                dt = t.get(); t.lap()
                print("Batch {}/{}, ms/batch: {}, loss: {:.5f}".format(i, num_batches, dt / (num_batches/10), tot_loss/(i)))
    else:
        loader = dataset.create_batches(batch_size=batch_size, shuffle=True)

        num_batches = len(dataset) // batch_size

        for batch, fixation in loader:


            loss, res, state = train_batch(ms, batch, fixation, loss_fn, state, scaler=scaler)
            tot_loss += loss

            if tot_res is None:
                tot_res = res
            else:
                tot_res += res

            if verbose and (i+1) % int(num_batches/10) == 0:
                dt = t.get(); t.lap()
                print("Batch {}/{}, ms/batch: {}, loss: {:.5f}".format(i, num_batches, dt / (num_batches/10), tot_loss/(i)))

    tot_loss /= num_batches
    tot_res /= num_batches
    
    
    print("Training loss: {:.8f}".format(tot_loss))

    if state is None:
        return tot_loss, tot_res, None
    else:
        return tot_loss, tot_res, state.detach()

def train(ms: ModelState,
          train_ds: Dataset,
          test_ds: Dataset,
          loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
          num_epochs: int = 1,
          batch_size: int = 32,
          sequence_length: int = 3,
          patience: int = 500,
          verbose = False,
          mnist=False,
          scaler=None,
          start_epoch=0):
    ms_name = ms.title.split('/')[-1]
    best_epoch = 0; tries = 0
    best_loss = sys.float_info.max
    best_network = None

    for epoch in range(start_epoch+1, num_epochs):
        print("Epoch {}, Lossfn {}".format(epoch, ms_name))

        train_loss, train_res, h = train_epoch(ms, train_ds, loss_fn, batch_size, sequence_length, verbose=verbose, mnist=mnist, scaler=scaler)
        ms.lr_schedule()

        test_loss, test_res = test_epoch(ms, test_ds, loss_fn, batch_size, sequence_length, mnist=mnist)
        wandb.log({'train_loss': train_loss,
                   'test_loss': test_loss})

        if (test_loss < best_loss):
            best_loss = test_loss
            tries = 0
        else:
            print("Loss did not improve from", best_loss)
            tries = tries + 1
            if (tries >= patience):
                print("Stopping early")
                break
        if epoch%50 == 0 and epoch > 0:
            ms.save(epoch)
