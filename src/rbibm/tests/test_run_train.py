from rbibm.runs.run_train import run_train
from rbi.loss import NLLLoss, NegativeElboLoss

import torch
def test_run_train_fKL(task, model):

    input_dim = task.input_dim
    output_dim = task.output_dim

    train_loader, validation_loader, test_loader = task.get_train_test_val_dataset(
        100, 100, 100, batch_size=50,
    )

    net = model(input_dim, output_dim)

    net, defense_model, train_loss, validation_loss, test_loss = run_train(
        task=task,
        model=net,
        loss_fn=NLLLoss,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        min_epochs=1,
        max_epochs=3,
    )

    assert all([torch.isfinite(t) for t in train_loss]), "Train loss should be finite"
    assert all([torch.isfinite(t) for t in validation_loss]), "Validation loss should be finite"
    assert torch.isfinite(test_loss), "Test loss should be finite"


def test_run_train_rKL(task_with_potential, model):

    input_dim = task_with_potential.input_dim
    output_dim = task_with_potential.output_dim


    train_loader, validation_loader, test_loader = task_with_potential.get_train_test_val_dataset(
        100, 100, 100, batch_size=50,
    )

    net = model(input_dim, output_dim, output_transform = torch.distributions.biject_to(task_with_potential.get_prior().support))

    net, defense_model, train_loss, validation_loss, test_loss = run_train(
        task=task_with_potential,
        model=net,
        loss_fn=NegativeElboLoss,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        loss_fn_hyper_parameters={"mc_samples": 1, "method": "joint_contrastive"},
        min_epochs=1,
        max_epochs=2,
    )

    assert all([torch.isfinite(t) for t in train_loss]), "Train loss should be finite"
    assert all([torch.isfinite(t) for t in validation_loss]), "Validation loss should be finite"
    assert torch.isfinite(test_loss), "Test loss should be finite"
