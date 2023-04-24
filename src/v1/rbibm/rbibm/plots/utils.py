
from rbibm.utils.utils_data import get_model_by_id, get_adversarial_examples_by_id,get_full_model_dataset
from rbibm.utils.get_task import get_task

def maybe_get_model_by_id(name, model_id, model):
    
    if isinstance(model_id, str):
        model = get_model_by_id(name, model_id)
    else:
        if model is None:
            raise ValueError("Either provide a model id or a model itself...")
    return model

def maybe_get_task_by_id(name, model_id, task):
    if isinstance(model_id, str) and isinstance(name, str):
        df = get_full_model_dataset(name)
        df = df[df.id == model_id]
        task = eval(df.params_task.to_list()[0])
        task_class = get_task(task["name"])
        return task_class(**task["params"])
    else:
        if task is None:
            raise ValueError("Either specifiy valid dataset and id or task")
        return task
        
    
def maybe_get_x_tilde_from_id(name, id_adversarial, x, theta, x_tilde):
    if isinstance(id_adversarial, str):
        x, theta, x_tilde = get_adversarial_examples_by_id(name, id_adversarial)
    else:
        if x is None or x_tilde is None:
            raise ValueError("Please provide either an id_adversarial or an x and x_tilde!")
    return x, theta, x_tilde
