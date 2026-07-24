from __future__ import annotations

def accelerator_info() -> dict:
    out={"available":False,"backend":"cpu","device":"cpu","name":"CPU","torch":None,"hip":None,"cuda":None}
    try:
        import torch
        out['torch']=torch.__version__
        out['hip']=getattr(torch.version,'hip',None)
        out['cuda']=getattr(torch.version,'cuda',None)
        if torch.cuda.is_available():
            out.update(available=True, device='cuda:0', name=torch.cuda.get_device_name(0), backend='rocm' if out['hip'] else 'cuda')
    except Exception as e:
        out['error']=str(e)
    return out

def resolve_device(requested: str='auto') -> str:
    r=str(requested).lower().strip()
    if r in ('cpu',): return 'cpu'
    if r in ('0','cuda','cuda:0','gpu'):
        return 'cuda:0' if accelerator_info()['available'] else 'cpu'
    if r == 'auto':
        return accelerator_info()['device']
    return requested
