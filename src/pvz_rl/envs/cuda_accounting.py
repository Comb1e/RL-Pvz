"""Read-only reward accounting hooks for the pinned CUDA simulator."""

from pvz_game.cuda import CudaBatch
from pvz_game.cuda.schema import EVENTS

ACCOUNTING_WIDTH = 10
EVENT_CODES = {name: index for index, (name, _) in enumerate(EVENTS)}


def accounting_kernel_source(source):
    replacements = {
        "double *f;": "double *f;\n  I *accounting;",
        "I ticks, I per_tick) {": "I ticks, I per_tick, I *accounting) {",
        "facts + i * 9};": f"facts + i * 9, accounting + i * {ACCOUNTING_WIDTH}}};",
        "    emit(5, a.id, source, dh, da);": "    if (source > 0) accounting[0] += dh + da;\n    emit(5, a.id, source, dh, da);",
        "    emit(4, id, source, h->sun - before, amount);": "    accounting[source == 0 ? 1 : 2] += h->sun - before;\n"
        "    emit(4, id, source, h->sun - before, amount);",
        "    *ne = 0;": f"    *ne = 0;\n    for (I j = 0; j < {ACCOUNTING_WIDTH}; j++) accounting[j] = 0;",
        "    if (DIAGNOSTIC) {": (
            f"    if (kind == {EVENT_CODES['SunProduced']}) accounting[3] += b;\n"
            f"    if (kind == {EVENT_CODES['PlantPlaced']}) {{ accounting[4] += PC[a]; accounting[8]++; }}\n"
            f"    if (kind == {EVENT_CODES['ZombieSpawned']}) accounting[5]++;\n"
            f"    if (kind == {EVENT_CODES['ZombieDefeated']}) accounting[6]++;\n"
            f"    if (kind == {EVENT_CODES['ZombieRemoved']}) accounting[7]++;\n"
            f"    if (kind == {EVENT_CODES['PlantRemoved']}) accounting[9]++;\n"
            "    if (DIAGNOSTIC) {"
        ),
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned CUDA accounting hook changed; verify the engine")
        source = source.replace(before, after)
    return source


class AccountingCudaBatch(CudaBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accounting = self.cp.zeros((self.n, ACCOUNTING_WIDTH), self.cp.int64)
        self.source = accounting_kernel_source(self.source)
        self.module = self.cp.RawModule(code=self.source, options=("--std=c++11", "--fmad=false"))
        self._step_kernel = self.module.get_function("step_games")
        self._mask = self.module.get_function("legal_masks")
        self._step = self._step_with_accounting

    def _step_with_accounting(self, grid, block, args):
        self._step_kernel(grid, block, (*args, self.accounting))
