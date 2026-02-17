import torch
import torch.nn.functional as F


class CifarInference:
    def __init__(self, model):
        self.model = model
    
    def _infer_basic(self, inputs):
        return self.model(inputs).clone()
    
    def _infer_mirror(self, inputs):
        return 0.5 * self.model(inputs) + 0.5 * self.model(inputs.flip(-1))
    
    def _infer_mirror_translate(self, inputs):
        logits = self._infer_mirror(inputs)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],  # up-left translation
            padded_inputs[:, :, 2:34, 2:34],  # down-right translation
        ]
        logits_translate_list = [self._infer_mirror(inputs_translate) 
                               for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate
    
    def infer(self, loader, tta_level=0):
        self.model.eval()
        test_images = loader.normalize(loader.images)
        
        infer_functions = [
            self._infer_basic,
            self._infer_mirror, 
            self._infer_mirror_translate
        ]
        infer_fn = infer_functions[tta_level]
        
        with torch.no_grad():
            return torch.cat([infer_fn(inputs) for inputs in test_images.split(2000)])
    
    def evaluate(self, loader, tta_level=0):
        logits = self.infer(loader, tta_level)
        return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate(model, loader, tta_level=0):
    inference = CifarInference(model)
    return inference.evaluate(loader, tta_level) 