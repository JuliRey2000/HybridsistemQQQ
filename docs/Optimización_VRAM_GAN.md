**Optimización: mover dataset completo a VRAM para eliminar overhead CPU→GPU**

**Contexto**

El entrenamiento WGAN-GP funciona correctamente pero el monitor de recursos muestra que la VRAM de la T4 solo usa **0.3 GB de 15 GB disponibles**. Con 2,445 ventanas de shape `(20, 1)` y sentimientos de shape `(768,)`, el dataset completo pesa menos de 10 MB. Actualmente cada batch se transfiere desde RAM (CPU) a VRAM (GPU) en cada paso del training loop, generando overhead innecesario en cada una de las \~1,200 transferencias por época (38 batches × n\_critic=5 \+ 38 pasos generator).

**Problema**

En la celda donde se construye `gan_loader`, los tensores se crean en CPU y el DataLoader los transfiere a GPU batch a batch durante el entrenamiento:

python  
gan\_ds     \= TensorDataset(torch.from\_numpy(gan\_windows), torch.from\_numpy(gan\_sents))  
gan\_loader \= TorchLoader(gan\_ds, batch\_size=64, shuffle=True, drop\_last=True)

**Solución**

Mover los tensores completos a VRAM antes de construir el dataset, para que el DataLoader sirva batches que ya viven en GPU sin ninguna transferencia en runtime:

python

\# Mover TODO el dataset a VRAM de una vez (\~10 MB total, cabe 1500x en los 15 GB disponibles)  
gan\_windows\_gpu \= torch.from\_numpy(gan\_windows).to(device)  
gan\_sents\_gpu   \= torch.from\_numpy(gan\_sents).to(device)

gan\_ds     \= TensorDataset(gan\_windows\_gpu, gan\_sents\_gpu)  
gan\_loader \= TorchLoader(  
    gan\_ds,  
    batch\_size=64,  
    shuffle=True,  
    drop\_last=True,  
    \# Sin pin\_memory ni num\_workers — los datos ya están en VRAM, no hay transferencia CPU→GPU  
)

**Dónde está el cambio**

Buscar en el notebook la celda que contiene `TensorDataset` y `TorchLoader` y que imprime `Ventanas GAN`, `Sentimiento GAN` y `Batches / época`. Es exactamente ahí donde se reemplazan las dos líneas de construcción del dataset y el loader por las de arriba.

**Resultado esperado**

El overhead de transferencia CPU→GPU desaparece completamente. Las 200 épocas deberían bajar de \~14 minutos a **2–4 minutos** totales, y el uso de VRAM debería subir de 0.3 GB a \~0.5 GB (los 10 MB del dataset más los pesos del modelo y los gradientes).

