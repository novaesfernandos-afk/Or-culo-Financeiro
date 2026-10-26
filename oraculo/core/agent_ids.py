"""Identificadores dos 18 agentes (usados nas permissões das skills)."""

CIIO = "A01"  # Chief Investment Intelligence Officer
ORQUESTRADOR = "A02"
ECON_GLOBAL = "A03"
ECON_BRASIL = "A04"
GEOPOLITICA = "A05"
COMMODITIES = "A06"
FUNDAMENTALISTA = "A07"
TECNICO = "A08"
SETORIAL = "A09"
FLUXO = "A10"
QUANT = "A11"
SAZONAL = "A12"
NOTICIAS = "A13"
RISCO = "A14"
AUDITOR = "A15"
COMPLIANCE = "A16"
CARTEIRA = "A17"
COMUNICADOR = "A18"

ALL = tuple(f"A{i:02d}" for i in range(1, 19))
