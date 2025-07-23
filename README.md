# AIETA - Hear-to-See
Repository per il codice associato al progetto per il corso di "Artificial Intelligence from Engineering to Arts" dell'Università degli Studi di Roma Tre.


# Descrizione
Proponiamo un sistema end-to-end che ascolta un racconto parlato di lunga durata e restituisce una sequenza di immagini in stile storyboard che ne evidenzia i punti 
di svolta chiave. La pipeline, orchestrata in ComfyUI, (i) rileva il parlato, (ii) separa la traccia vocale da musica o rumori ambientali, quando necessario, (iii) 
identifica i picchi che segnano momenti semanticamente rilevanti e (iv) ciascuno di questi segmenti viene trascritto in testo con il modello Whisper di OpenAI e poi 
associato a un’immagine illustrativa tramite un modello di diffusione commerciale. I risultati sperimentali dimostrano un solido allineamento visivo tra le immagini 
generate e gli eventi narrativi corrispondenti: parte della nostra valutazione ha incluso test su fiabe classiche e narrazioni cinematografiche iconiche, presentate 
come clip audio continue di due-cinque minuti.
