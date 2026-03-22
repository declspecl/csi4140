import torch


def _set_training_mode(model, mode: bool) -> None:
    for layer in model.layers:
        if hasattr(layer, "training"):
            layer.training = mode


def _zero_grad(optimizer) -> None:
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    else:
        for _, param in optimizer.params:
            param.grad = None


def train_epoch(model, train_loader, loss_fn, optimizer, device="cpu") -> tuple[float, float]:
    """
    Runs one full pass over train_loader.
    Returns: (avg_loss, accuracy) for the epoch.
    """
    _set_training_mode(model, True)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        probs = model.forward(images)

        loss = loss_fn.calculate_loss(probs, labels)
        total_loss += loss.item()

        preds = probs.argmax(dim=0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        _zero_grad(optimizer)

        grad = loss_fn.calculate_gradient(probs, labels)
        model.backward(grad)

        optimizer.step()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device="cpu") -> tuple[float, float]:
    """
    Evaluates on test_loader without updating weights.
    Returns: (avg_loss, accuracy).
    """
    _set_training_mode(model, False)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            probs = model.forward(images)

            loss = loss_fn.calculate_loss(probs, labels)
            total_loss += loss.item()

            preds = probs.argmax(dim=0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    epochs: int = 50,
    device: str = "cpu",
    checkpoint_path: str = "best_model.pt",
) -> dict:
    """
    Runs full training for `epochs` epochs.
    Returns history dict with train_loss, train_acc, test_loss, test_acc.
    Saves model layers' state when test_acc improves.
    Prints per-epoch summary.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_test_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
            f"test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {id(p): p.data.clone() for _, p in optimizer.params}
            torch.save(best_state, checkpoint_path)

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        for _, p in optimizer.params:
            p.data.copy_(best_state[id(p)])

    return history
