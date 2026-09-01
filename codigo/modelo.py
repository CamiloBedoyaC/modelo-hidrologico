"""
Implementación del modelo hidrológico conceptual de dos tanques.
Ecuaciones según el enunciado de la tarea.

Idea general:
- Entra precipitación diaria P (mm/d).
- Se descuenta evapotranspiración (ET) para obtener precipitación neta (Pneta).
- Pneta alimenta un primer “tanque” (S1) que genera salidas (caudal) y transfiere agua al segundo tanque (S2).
- El segundo tanque genera caudal más lento (baseflow) y también puede desbordarse.
- Todo está en mm/d (lámina), lo que facilita revisar el cierre del balance hídrico.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TwoTankParams:
    """
    Contenedor de parámetros del modelo (todos escala diaria).

    - ETc: evapotranspiración “base” (mm/d).
    - beta: cuánto aumenta ET con la lluvia diaria (adimensional).
    - alpha1: fracción de precipitación neta que se va directo a escorrentía (adimensional).
    - D1: capacidad máxima del tanque 1 (mm).
    - k1: fracción diaria que drena del tanque 1 como flujo lento (adimensional, 0-1).
    - alpha2: fracción del drenaje del tanque 1 que se vuelve “rápido” hacia el cauce (adimensional).
    - D2: capacidad máxima del tanque 2 (mm).
    - k2: fracción diaria que drena del tanque 2 como flujo lento/base (adimensional, 0-1).
    """
    ETc: float
    beta: float
    alpha1: float
    D1: float
    k1: float
    alpha2: float
    D2: float
    k2: float


def two_tank_run(precip_mm: np.ndarray, params: TwoTankParams, return_states: bool = False):
    """
    Ejecuta el modelo de dos tanques.

    Parámetros
    ----------
    precip_mm : np.ndarray
        Serie diaria de precipitación P en mm/día.
    params : TwoTankParams
        Parámetros del modelo (ETc, beta, alpha1, D1, k1, alpha2, D2, k2).
    return_states : bool
        Si es True, además del caudal simulado devuelve variables internas
        para auditar el balance hídrico.

    Retorna
    -------
    Q : np.ndarray
        Caudal simulado en mm/día (lámina equivalente sobre la cuenca).

    Si return_states=True retorna también:
    - ET_arr : ET usada (mm/d)
    - S1_arr : almacenamiento tanque 1 (mm)
    - S2_arr : almacenamiento tanque 2 (mm)
    - dS_arr : cambio diario total de almacenamiento Δ(S1+S2) (mm/d)

    Nota importante (balance):
    - El enunciado define una ET_pot = ETc + beta*P.
    - Pero físicamente, no puede evaporarse más agua que la que “llega” ese día.
      Por eso se limita ET al máximo P (ET = min(ET_pot, P)).
    - Con esa convención, Pneta = max(P - ET_pot, 0) y ET_usada = P - Pneta,
      y así el balance puede cerrar sin “crear” o “destruir” agua.
    """
    
    # Número de días a simular.
    n = len(precip_mm)
    # Arreglos donde vamos guardando las salidas día a día.
    Q = np.zeros(n, dtype=float)
    ET_arr = np.zeros(n, dtype=float)
    S1_arr = np.zeros(n, dtype=float)
    S2_arr = np.zeros(n, dtype=float)
    dS_arr = np.zeros(n, dtype=float)
    # Estados iniciales (condición inicial del sistema).
    # Aquí se parte en cero, y normalmente luego se usa un warm-up para “estabilizar”.
    S1 = 0.0
    S2 = 0.0

    # “Desempaquetamos” parámetros para que el loop quede más legible.
    ETc, beta, alpha1, D1, k1, alpha2, D2, k2 = (
        params.ETc,
        params.beta,
        params.alpha1,
        params.D1,
        params.k1,
        params.alpha2,
        params.D2,
        params.k2,
    )

    # Recorremos día por día y actualizamos los tanques.
    for t in range(n):
        P = precip_mm[t]
        # Guardamos estados anteriores para calcular ΔS al final del día.
        S1_prev, S2_prev = S1, S2
        # 1) Evapotranspiración potencial según el enunciado.
        ET_pot = ETc + beta * P
        # 2) Precipitación neta: lo que queda después de “intentar” evaporar.
        # Si ET_pot supera a P, no queda nada neto: Pneta = 0.
        Pneta = max(P - ET_pot, 0.0)
        # 3) ET usada realmente:
        # es lo que se “consumió” del P del día. Esto equivale a min(ET_pot, P).
        ET = P - Pneta  # ET usada (limitada por P)

        # -----------------------------
        # Tanque 1
        # -----------------------------

        # 4) Escorrentía directa (rápida) desde Pneta.
        # Si alpha1=0, no hay aporte directo; si alpha1 es alto, el modelo reacciona más rápido.
        Qdirecta = alpha1 * Pneta
        # 5) Lo que NO fue directo entra al almacenamiento S1.
        S1_temp = S1 + (1 - alpha1) * Pneta
        # 6) Desborde del tanque 1 si se supera su capacidad D1.
        # El exceso se va como caudal (Qdesborde1).
        Qdesborde1 = 0.0
        if S1_temp > D1:
            Qdesborde1 = S1_temp - D1
            S1_temp = D1
        # 7) Salida lenta del tanque 1: proporción k1 del almacenamiento.
        # Representa drenaje/infiltración lenta hacia el sistema.
        Qlento1 = k1 * S1_temp
        # 8) Actualizamos S1 restando lo que salió.
        S1 = S1_temp - Qlento1

        # -----------------------------
        # Tanque 2
        # -----------------------------

        # 9) Parte del drenaje del tanque 1 puede convertirse en “rápido” hacia el cauce.
        # (En tu auditoría, alpha2 quedó en 0 muchas veces, entonces este componente desaparece.)
        Qrapido2 = alpha2 * Qlento1
        # 10) El resto del flujo lento del tanque 1 alimenta el tanque 2 como almacenamiento.
        S2_temp = S2 + (1 - alpha2) * Qlento1
        # 11) Desborde del tanque 2 si se supera su capacidad D2.
        Qdesborde2 = 0.0
        if S2_temp > D2:
            Qdesborde2 = S2_temp - D2
            S2_temp = D2
        # 12) Salida lenta del tanque 2: k2 del almacenamiento (baseflow típico).
        Qlento2 = k2 * S2_temp
        # 13) Actualizamos S2 restando lo que salió.
        S2 = S2_temp - Qlento2

        # -----------------------------
        # Caudal total del día y estados
        # -----------------------------

        # 14) Caudal total simulado = suma de componentes (todos en mm/d).
        Q[t] = Qdirecta + Qdesborde1 + Qrapido2 + Qdesborde2 + Qlento2
        # 15) Guardamos series para diagnóstico.
        ET_arr[t] = ET
        S1_arr[t] = S1
        S2_arr[t] = S2
        # 16) Cambio de almacenamiento total ΔS = (S1+S2) - (S1_prev+S2_prev).
        # Útil para verificar balance hídrico.
        dS_arr[t] = (S1 - S1_prev) + (S2 - S2_prev)

    if return_states:
        return Q, ET_arr, S1_arr, S2_arr, dS_arr
    return Q


def default_params() -> TwoTankParams:
    """
    Devuelve un set de parámetros “por defecto”.

    - Son valores razonables para correr el modelo sin calibrar.
    - Deben estar dentro de los rangos del enunciado/Tabla.
    - Sirven como punto de partida antes de calibración.

    Nota: estos valores NO garantizan buen NSE; solo garantizan que el modelo corre.
    """

    return TwoTankParams(
        ETc=2.0,
        beta=0.2,
        alpha1=0.2,
        D1=150.0,
        k1=0.08,
        alpha2=0.3,
        D2=400.0,
        k2=0.03,
    )
