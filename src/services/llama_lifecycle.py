import os
import subprocess
import time
from pathlib import Path

import httpx


class LlamaLifecycleManager:
    """Gerencia o ciclo de vida do llama-server sob demanda para preservar VRAM durante o treino."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        project_dir: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.project_dir = Path(project_dir or Path.home() / "workspace" / "llama.cpp")
        self._server_proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        """Verifica se o servidor local está ativo e respondendo."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.base_url}/models")
                return res.status_code == 200
        except Exception:
            return False

    def stop_server(self) -> None:
        """Encerra qualquer instância ativa do llama-server para liberar VRAM."""
        os.system("pkill -9 -f 'llama-server' >/dev/null 2>&1")
        time.sleep(1.0)

    def start_server(self, timeout: int = 30) -> bool:
        """Inicia o llama-server via subprocess em background com o comando oficial."""
        if self.is_running():
            return True

        server_bin = self.project_dir / "build" / "bin" / "llama-server"
        if not server_bin.exists():
            return False

        cmd = [
            str(server_bin),
            "-hf",
            "unsloth/gemma-4-E4B-it-GGUF",
            "-m",
            "gemma-4-E4B-it-Q4_K_M.gguf",
            "-ngl",
            "999",
            "-c",
            "8192",  # 8k contexto para resposta rápida e leve
            "-t",
            "8",
            "--flash-attn",
            "on",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]

        env = os.environ.copy()
        env["LLAMA_CACHE"] = "models"

        self._server_proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Polling até ficar pronto
        start_t = time.time()
        while time.time() - start_t < timeout:
            if self.is_running():
                return True
            time.sleep(1.0)
        return False
