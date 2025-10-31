# after constructing model/opt/scaler
from pathlib import Path
save_checkpoint(Path("model/checkpoints/last_model.pt"),
                model=model, optimizer=opt, scaler=scaler,
                epoch=1, step=123, optimizer_step=45, cfg={"hello":"world"})
e, s, os = load_checkpoint(Path("model/checkpoints/last_model.pt"), model=model)
print(e, s, os)  # expect: 1 123 45
