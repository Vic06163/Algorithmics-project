# Dictionnaire permettant de totaliser le nombre de fois qu'un des 16 sommets est visité
dico_verification = {
    1: 0, 
    2: 0, 
    3: 0, 
    4: 0, 
    5: 0
}

# Listes de cycles proposés en solution
cycles = [
    [2, 1, 3, 2],
    [2, 5, 4, 2]
]

depot = 2 # Sommet de départ et d'arrivée (dépôt)
is_valid = True # La solution est-elle valide ou non

for cycle in cycles:
    # On vérifie que le cycle commence et se termine bien au dépôt
    if cycle[0] != depot or cycle[-1] != depot:
        is_valid = False
        break

    cycle.pop(0) # On retire le premier élément du cycle (qui est le dépôt)
    cycle.pop(-1) # On retire le dernier élément du cycle (qui est le dépôt)

    # On vérifie que le cycle ne contient pas de doublons
    for sommet in cycle:
        if dico_verification[sommet] == 1:
            is_valid= False
            break
        else:
            dico_verification[sommet] = 1

if is_valid:
    for sommet in dico_verification:
        if sommet == depot:
            continue

        # On vérifie que chaque sommet est visité une et une seule fois
        if dico_verification[sommet] == 0:
            is_valid = False
            break

if(is_valid):
    print("La solution proposée est valide")
else:
    print("La solution proposée n'est pas valide")