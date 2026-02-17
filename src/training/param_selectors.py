def select_parameters(model, config, exclude_params=None):
    """Select parameters from model based on config.
    
    Args:
        model: The model to select parameters from
        config: Selector configuration dict
        exclude_params: Set of parameter IDs to exclude (to avoid overlaps)
    """
    if exclude_params is None:
        exclude_params = set()
    
    selector_type = config.get('type', 'all')
    
    if selector_type == 'all':
        return [p for p in model.parameters() 
                if p.requires_grad and id(p) not in exclude_params]
    
    elif selector_type == 'name_match':
        pattern = config['pattern']
        return [p for n, p in model.named_parameters() 
                if n == pattern and p.requires_grad and id(p) not in exclude_params]
    
    elif selector_type == 'name_contains':
        pattern = config['pattern']
        return [p for n, p in model.named_parameters() 
                if pattern in n and p.requires_grad and id(p) not in exclude_params]
    
    elif selector_type == 'name_contains_multiple':
        patterns = config['patterns']
        return [p for n, p in model.named_parameters() 
                if any(pattern in n for pattern in patterns) and p.requires_grad and id(p) not in exclude_params]
    
    elif selector_type == 'shape_match':
        ndim = config['ndim']
        return [p for p in model.parameters() 
                if len(p.shape) == ndim and p.requires_grad and id(p) not in exclude_params]
    
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")