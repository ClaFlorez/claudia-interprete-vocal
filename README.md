# CLAUD·IA — Intérprete vocal trilingüe

Trabajo práctico de aprendizaje automático (*Machine Learning*). El proyecto traduce frases cortas habladas entre francés, inglés y español y genera un audio que se puede escuchar.

## Programa

El notebook de Python utiliza el Transformer preentrenado SeamlessM4T v2.
Carga un fragmento de audio, lo prepara en mono a 16 kHz, selecciona el idioma de destino y permite escuchar la traducción generada.

El ajuste fino supervisado (*fine-tuning*) piloto se centró en la traducción de voz en español a texto en inglés. La síntesis de voz utiliza los parámetros preentrenados. También se realizaron pruebas de francés con el modelo preentrenado.

## Datos y evaluación

**FLEURS** significa *Few-shot Learning Evaluation of Universal Representations of Speech*. Este conjunto de datos publicado por Google contiene grabaciones de frases equivalentes en varios idiomas. Para el ajuste fino de nuestro Transformer, cada ejemplo relaciona una grabación en español con su traducción escrita al inglés. [Fuente de los datos](https://huggingface.co/datasets/google/fleurs)

- **Entrenamiento y validación** — Grabaciones en español y sus correspondientes traducciones escritas al inglés, procedentes de particiones distintas de FLEURS.
- **Prueba final** — Seis frases originales en español grabadas con voz normal y voz baja, para un total de doce audios. Se utilizaron los mismos archivos para comparar el modelo preentrenado con el modelo ajustado.
- **Resultado** — En ambas versiones del modelo se conservó el sentido en las seis grabaciones con voz normal y en cinco de las seis grabaciones con voz baja. Una prueba independiente midió el tiempo de generación en CPU y en una GPU T4. [Protocolo, resultados y capturas](INSTALLATION_ET_DEPANNAGE.md)

## Entregables

- [x] Notebook de demostración e instrucciones de ejecución.
- [x] Código de ajuste fino y protocolo de la prueba piloto.
- [x] Ejemplo de demostración, fuente de datos y resultados medidos.
- [ ] Diapositivas y resumen de una a dos páginas.

## Fuentes

- [SeamlessM4T v2 — Meta](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [FLEURS — Google](https://huggingface.co/datasets/google/fleurs)
- [Guía de ajuste fino — Meta](https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/cli/m4t/finetune)
