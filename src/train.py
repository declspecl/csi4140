import torch


def _collect_params(model):
    params = []
    for i, layer in enumerate(model.layers):
        for name, param in layer.parameters().items():
            key = name.value if hasattr(name, "value") else name
            params.append(((i, key), param))
    return params


def train_epoch(model, train_loader, loss_fn, optimizer, device="cpu", iteration_losses=None) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # FIX: nn.Parameter has requires_grad=True by default, so PyTorch builds
        # an autograd graph for every operation (einsum, mm, etc.) even though we
        # compute gradients manually. This was causing ~14GB of GPU memory usage
        # from the stored computation graph alone. no_grad disables this.
        with torch.no_grad():
            probs = model.forward(images)

            loss = loss_fn.calculate_loss(probs, labels)
            loss_val = loss.item()
            total_loss += loss_val

            if iteration_losses is not None:
                iteration_losses.append(loss_val)

            preds = probs.argmax(dim=0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()

            grad = loss_fn.calculate_gradient(probs, labels)
            model.backward(grad)

            optimizer.step()

        if batch_idx % 50 == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)} - loss: {loss_val:.4f}", flush=True)

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device="cpu") -> tuple[float, float]:
    model.eval()
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
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "iteration_loss": [],
    }
    best_test_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            iteration_losses=history["iteration_loss"],
        )
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
            best_state = {key: p.data.cpu().clone() for key, p in _collect_params(model)}
            torch.save(best_state, checkpoint_path)

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        param_map = {key: p for key, p in _collect_params(model)}
        for key, data in best_state.items():
            param_map[key].data.copy_(data.to(param_map[key].data.device))

    return history
