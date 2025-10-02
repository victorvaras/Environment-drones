

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 

# 5G NR (PUSCH)
from sionna.phy.nr.pusch_config import PUSCHConfig
from sionna.phy.nr.pusch_transmitter import PUSCHTransmitter
from sionna.phy.nr.pusch_receiver import PUSCHReceiver

# MIMO
from sionna.phy.mimo.stream_management import StreamManagement
from sionna.phy.mimo.detection import KBestDetector, LinearDetector  # si usas K-best/LMMSE

# Canal OFDM
from sionna.phy.channel.ofdm_channel import OFDMChannel

# Utilidades (ratio Eb/N0 -> N0)
from sionna.phy.utils.misc import ebnodb2no


# ====== Upload link level ======

import tensorflow as tf

def _repeat_to_batch_exact(x, batch_size: tf.Tensor):
    b0 = tf.shape(x)[0]
    def rep1():
        return tf.repeat(x, repeats=batch_size, axis=0)
    def repN():
        reps = tf.cast(tf.math.ceil(tf.cast(batch_size, tf.float32)/tf.cast(b0, tf.float32)), tf.int32)
        xr = tf.tile(x, tf.concat([[reps], tf.ones([tf.rank(x)-1], tf.int32)], axis=0))
        return xr[:batch_size]
    return tf.cond(tf.equal(b0, 1), rep1, repN)

def cir_sampler_core(a0: tf.Tensor, t0: tf.Tensor,
                     batch_size: tf.Tensor,
                     num_ofdm_symbols: tf.Tensor,
                     sampling_frequency: tf.Tensor):
    # Ajuste de batch
    a = _repeat_to_batch_exact(a0, batch_size)   # [B,R,Rant,TX,TXant,P,T0]
    tau = _repeat_to_batch_exact(t0, batch_size) # [B,R,TX,P]
    # Ajuste temporal
    T0 = tf.shape(a)[6]
    a = tf.cond(tf.equal(T0, num_ofdm_symbols),
                lambda: a,
                lambda: tf.repeat(a, repeats=num_ofdm_symbols, axis=6))
    return a, tau

# --- Wrapper tolerante a si TF le pasa un "self" fantasma (4 args) o no (3 args)
def cir_sampler_wrapper(*args):
    """
    Admite:
      (batch_size, num_ofdm_symbols, sampling_frequency)
    o   (self, batch_size, num_ofdm_symbols, sampling_frequency)
    Usa tensores globales A0/T0 que DEBES cargar antes de construir el canal.
    """
    a0 = _A0_GLOB
    t0 = _T0_GLOB
    if a0 is None or t0 is None:
        raise RuntimeError("A0/T0 globales no inicializados")
    # Desempaquetar args
    if len(args) == 3:
        batch_size, num_ofdm_symbols, sampling_frequency = args
    elif len(args) == 4:
        _, batch_size, num_ofdm_symbols, sampling_frequency = args
    else:
        raise TypeError(f"cir_sampler_wrapper recibió {len(args)} args, esperaba 3 o 4")
    return cir_sampler_core(a0, t0, batch_size, num_ofdm_symbols, sampling_frequency)

# Buffers globales que alimentan al wrapper
_A0_GLOB = None   # tf.complex64 [B0,R,Rant,TX,TXant,P,T0]
_T0_GLOB = None   # tf.float32  [B0,R,TX,P]





def make_fixed_cir_sampler(a_np, tau_np):
    """
    Factory que devuelve una FUNCIÓN LIBRE con firma:
      sampler(batch_size, num_ofdm_symbols, sampling_frequency) -> (a, tau)
    a_np: [B0,R,Rant,TX,TXant,P,T0]  (numpy, complex64)
    tau_np: [B0,R,TX,P]              (numpy, float32)
    """


    # Congelamos tensores base UNA vez aquí; el sampler devuelto no tiene estado.
    a0 = tf.convert_to_tensor(a_np,  tf.complex64)
    t0 = tf.convert_to_tensor(tau_np, tf.float32)

    def sampler(batch_size, num_ofdm_symbols, sampling_frequency):
        # Asegurar batch EXACTO
        a = _repeat_to_batch_exact(a0, batch_size)   # [B, R, Rant, TX, TXant, P, T0]
        tau = _repeat_to_batch_exact(t0, batch_size) # [B, R, TX, P]

        # Asegurar eje temporal T = num_ofdm_symbols (si T0 != Nsym, repetimos sobre T)
        T0 = tf.shape(a)[6]
        a = tf.cond(tf.equal(T0, num_ofdm_symbols),
                    lambda: a,
                    lambda: tf.repeat(a, repeats=num_ofdm_symbols, axis=6))
        return a, tau

    return sampler




# Pushh link 5k

    def get_cir_current_positions(self, scs_hz: float):
        """
        Obtiene (a, tau) para el estado ACTUAL de la escena (pos del dron/TX y RX).
        Devuelve:
        a   : [1, RX, RX_ant, TX, TX_ant, P, T] complejo64
        tau : [1, RX, TX, P] float32, en segundos
        """
        paths = self._paths()

        a_raw, tau_raw = paths.cir(out_type="numpy")  # a: complejo, tau: float [s]
        a = np.asarray(a_raw, dtype=np.complex64)
        tau = np.asarray(tau_raw, dtype=np.float32)

        # ---- Normalización de 'a' ----
        # Esperamos finalmente: [1, R, Rant, TX, TXant, P, T]
        if a.ndim == 6:
            # Típico RT: [R, Rant, TX, TXant, P, T] -> añade batch
            a = a[np.newaxis, ...]
        elif a.ndim == 5:
            # [R, Rant, TX, TXant, P] -> añade T=1 y batch
            a = a[..., np.newaxis]     # T=1
            a = a[np.newaxis, ...]     # B=1
        elif a.ndim == 7:
            # Ya tiene batch/tiempo
            pass
        else:
            raise ValueError(f"Forma inesperada de a (esperaba 5/6/7 dims), obtuve {a.shape}")

        # ---- Normalización de 'tau' ----
        # Queremos finalmente: [1, R, TX, P]
        if tau.ndim == 3:
            # [R, TX, P] -> añade batch
            tau = tau[np.newaxis, ...]
        elif tau.ndim == 4:
            # [B, R, TX, P] -> ok
            pass
        elif tau.ndim == 5:
            # Caso que te aparece: [R, Rant(=1), TX(=1), TXant(=1), P]
            # Quitamos ejes de antena (y TX si viniera como 1 redundante)
            R, Rant, TX, TXant, P = tau.shape
            # Asegurarnos que los ejes de antena son 1
            assert Rant == 1 and TXant == 1, f"tau con antenas >1 no soportado: {tau.shape}"
            # Si TX==1, es equivalente a [R, TX, P] con TX=1
            tau = tau.reshape(R, 1, P)         # [R, TX, P]
            tau = tau[np.newaxis, ...]         # [1, R, TX, P]
        else:
            raise ValueError(f"Forma inesperada de tau (esperaba 3/4/5 dims), obtuve {tau.shape}")

        # ---- Chequeo de consistencia ----
        # a: [B,R,Rant,TX,TXant,P,T] ; tau: [B,R,TX,P]
        assert a.shape[0] == tau.shape[0], f"Batch mismatch: {a.shape} vs {tau.shape}"
        assert a.shape[1] == tau.shape[1], f"R mismatch: {a.shape} vs {tau.shape}"
        assert a.shape[3] == tau.shape[2], f"TX mismatch: {a.shape} vs {tau.shape}"
        assert a.shape[5] == tau.shape[3], f"P mismatch: {a.shape} vs {tau.shape}"

        return a, tau



    def simulate_pusch_bler(self,
                            ebno_db: float = 10.0,
                            num_blocks: int = 500,
                            perfect_csi: bool = False,
                            detector: str = "lmmse",
                            num_prb: int = 16,
                            scs_hz: float = 30e3,
                            mcs_index: int = 14,
                            mcs_table: int = 1,
                            num_layers: int = 1,
                            batch_size: int = 8):
        """
        Simulación E2E PUSCH con el estado actual de la escena (sin dataset).
        Retorna: dict con BER, BLER, bloques_OK/total, etc.
        """
        assert self.scene is not None and self._solver is not None, "Llama a build_scene() antes."
        assert self.rx_list, "Llama a attach_receivers(...) antes."

        # 1) CIR desde el RT actual (punto fijo)
        a_np, tau_np = self.get_cir_current_positions(scs_hz)  # [1,R,Rant,TX,TXant,P,1], [1,R,TX,P]

        # 2) Config PUSCH (coherente con #antenas TX)
        num_tx     = int(a_np.shape[3])
        num_tx_ant = int(a_np.shape[4])

        pc = PUSCHConfig()
        pc.carrier.subcarrier_spacing = scs_hz / 1e3    # kHz
        pc.carrier.n_size_grid = int(num_prb)
        pc.num_layers = num_layers
        pc.tb.mcs_index = mcs_index
        pc.tb.mcs_table = mcs_table

        # --- clave: precoding según #puertos ---
        if num_tx_ant >= 2:
            pc.num_antenna_ports = num_tx_ant
            pc.precoding = "codebook"
            pc.tpmi = 1
            pc.dmrs.dmrs_port_set = list(range(num_layers))
        else:
            # 1 puerto -> SIN codebook
            pc.num_antenna_ports = 1
            pc.precoding = "non-codebook"   # <-- este es el valor válido en Sionna 1.2
            # tpmi no aplica en non-codebook
            pc.dmrs.dmrs_port_set = [0]

        # DM-RS básicos
        pc.dmrs.config_type = 1
        pc.dmrs.length = 1
        pc.dmrs.additional_position = 1
        pc.dmrs.num_cdm_groups_without_data = 2
        # asigna puertos DM-RS (si solo hay 1 puerto físico, usa siempre 0)
        pc.dmrs.dmrs_port_set = list(range(num_layers)) if pc.num_antenna_ports > 1 else [0]

        # Clonar por transmisor y asignar puertos DM-RS no solapados
        pusch_cfgs = [pc]
        for i in range(1, num_tx):
            pc_i = pc.clone()
            if pc.precoding == "codebook":
                pc_i.dmrs.dmrs_port_set = list(range(i * num_layers, (i + 1) * num_layers))
            else:
                pc_i.dmrs.dmrs_port_set = [0]  # 1 puerto físico -> todos usan 0
            pusch_cfgs.append(pc_i)


        tx = PUSCHTransmitter(pusch_cfgs, output_domain="freq")

        # 3) Detector/Receiver
        R = int(a_np.shape[1])                  # número de receptores del CIR
        rx_tx_assoc = np.ones([R, num_tx], bool)
        sm = StreamManagement(rx_tx_assoc, num_layers)

        assert detector in ["lmmse", "kbest"], "Detector no soportado"
        if detector == "lmmse":
            det = LinearDetector(equalizer="lmmse",
                                output="bit",
                                demapping_method="maxlog",
                                resource_grid=tx.resource_grid,
                                stream_management=sm,
                                constellation_type="qam",
                                num_bits_per_symbol=pc.tb.num_bits_per_symbol)
        else:
            det = KBestDetector(output="bit",
                                num_streams=num_tx * num_layers,
                                k=64,
                                resource_grid=tx.resource_grid,
                                stream_management=sm,
                                constellation_type="qam",
                                num_bits_per_symbol=pc.tb.num_bits_per_symbol)

        if perfect_csi:
            rx = PUSCHReceiver(tx, mimo_detector=det, input_domain="freq", channel_estimator="perfect")
        else:
            rx = PUSCHReceiver(tx, mimo_detector=det, input_domain="freq")


        # 4) Canal OFDM a partir de (a,tau) EN MEMORIA — API Sionna 1.2 (sampler)
        # Dummy requerido por firma (no se usa si pasamos cir_sampler)

        global _A0_GLOB, _T0_GLOB
        _A0_GLOB = tf.convert_to_tensor(a_np,  tf.complex64)
        _T0_GLOB = tf.convert_to_tensor(tau_np, tf.float32)

        class _UnusedModel(tf.Module):
            def __call__(self, x, no):
                return x, no

        print("cir_sampler =", cir_sampler_wrapper, 
            "module:", getattr(cir_sampler_wrapper, "__module__", None),
            "qualname:", getattr(cir_sampler_wrapper, "__qualname__", None))

        ch = OFDMChannel(
            channel_model=_UnusedModel(),
            resource_grid=tx.resource_grid,
            normalize_channel=True,
            return_channel=True,
            cir_sampler=cir_sampler_wrapper,   # ← tolera 3 o 4 args
        )


        # 5) Monte Carlo de bloques
        blocks_ok = 0
        blocks_total = 0
        bit_errs = 0
        bits_total = 0

        print(f"Simulando {num_blocks} bloques PUSCH, batch={batch_size}, detector={detector}, perfect_csi={perfect_csi}")


        def _one(batch_size_tf, ebno_db_tf):
            x, b = tx(batch_size_tf)
            no = ebnodb2no(ebno_db_tf, tx._num_bits_per_symbol, tx._target_coderate, tx.resource_grid)
            y, h = ch(x, no)
            if perfect_csi:
                b_hat = rx(y, no, h)
            else:
                b_hat = rx(y, no)
            return b, b_hat

        while blocks_total < num_blocks:
            b, b_hat = _one(tf.constant(int(batch_size), tf.int32),
                            tf.constant(float(ebno_db), tf.float32))
            # -> [B, TX, tb_size]
            b_np     = tf.cast(b, tf.int8).numpy()
            b_hat_np = tf.cast(b_hat, tf.int8).numpy()

            # BER
            bit_errs += int(np.sum(b_np != b_hat_np))
            bits_total += int(np.prod(b_np.shape))

            # BLER (bloque OK si todos los bits por TX están correctos)
            ok_mask = np.all(b_np == b_hat_np, axis=2)  # [B, TX]
            blocks_ok += int(np.sum(ok_mask))
            blocks_total += int(ok_mask.size)

        ber  = bit_errs / max(1, bits_total)
        bler = 1.0 - (blocks_ok / max(1, blocks_total))

        return {
            "ebno_db": float(ebno_db),
            "num_blocks": int(blocks_total),
            "blocks_ok": int(blocks_ok),
            "bler": float(bler),
            "ber": float(ber),
            "detector": detector,
            "perfect_csi": bool(perfect_csi),
            "num_tx": int(a_np.shape[3]),
            "num_tx_ant": int(a_np.shape[4]),
            "num_rx": int(a_np.shape[1]),
        }