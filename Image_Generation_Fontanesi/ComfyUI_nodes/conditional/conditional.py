

class ConditionalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # valore generico da confrontare, può essere uscita di un altro nodo
            "compare_value": ("INT",),
            # soglia da utilizzare, fissa o proveniente da altro nodo
            "threshold": ("INT", {"default": 0.0}),
            # operatore di confronto
            "operator": (["<", ">", "==", "<=", ">=", "!="], {}),
            # input generico da passare se la condizione è vera
            "input": ("FILE",)
        }}
    CATEGORY = "control"
    # Restituisce lo stesso tipo dell'input
    RETURN_TYPES = ("FILE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "check"

    def check(self, compare_value, threshold, operator, input):
        # Esegue confronto tra valori generici convertiti a stringa
        try:
            expr = f"{compare_value} {operator} {threshold}"
            result = eval(expr)
        except Exception as e:
            raise RuntimeError(f"Error evaluating condition: {e}")
        if not result:
            # blocca l'esecuzione se condizione falsa
            raise RuntimeError(f"Il seguente file audio non contiene traccia vocale di parlato.")
        # passa il secondo input se condizione vera
        return (input,)

    @classmethod
    def IS_CHANGED(cls, compare_value, threshold, operator, input):
        return f"{compare_value}-{threshold}-{operator}"
