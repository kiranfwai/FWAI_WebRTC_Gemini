"""
Robust Audio Pipeline for WhatsApp <-> Gemini Live

This module handles all audio format conversions, resampling, and normalization.
Specifically designed to handle WhatsApp's quirky audio formats.
"""

import numpy as np
from typing import Optional, Tuple
from loguru import logger


class AudioPipeline:
    """
    Unified audio processing pipeline
    
    Handles:
    - Format detection and conversion
    - Intelligent amplification
    - Resampling between different sample rates
    - Noise gating
    """
    
    def __init__(self):
        # Audio level statistics for monitoring
        self.input_frame_count = 0
        self.output_frame_count = 0
        
        # Audio quality metrics
        self.avg_input_level = 0
        self.avg_output_level = 0
        
    def process_webrtc_to_gemini(
        self,
        audio_data: np.ndarray,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        apply_gain: bool = True
    ) -> Tuple[bytes, dict]:
        """
        Process audio from WebRTC (WhatsApp) for Gemini Live
        
        Args:
            audio_data: Audio as numpy array (any format)
            input_sample_rate: Sample rate of input audio
            output_sample_rate: Target sample rate for Gemini (16kHz)
            apply_gain: Whether to apply smart amplification
            
        Returns:
            Tuple of (PCM bytes, metadata dict)
        """
        self.input_frame_count += 1
        
        metadata = {
            'frame_num': self.input_frame_count,
            'input_shape': audio_data.shape,
            'input_dtype': str(audio_data.dtype),
        }
        
        # Step 1: Ensure mono audio
        audio = self._ensure_mono(audio_data)
        
        # Step 2: Convert to float32 for processing
        audio_float = self._to_float32(audio)
        
        # Step 3: Measure input level
        input_level = self._calculate_rms(audio_float)
        metadata['input_level'] = input_level
        
        # Step 4: Smart amplification for WhatsApp's quiet audio
        if apply_gain:
            audio_float, gain_applied = self._apply_smart_gain(audio_float)
            metadata['gain_applied'] = gain_applied
        else:
            metadata['gain_applied'] = 1.0
        
        # Step 5: Resample if needed
        if input_sample_rate != output_sample_rate:
            audio_float = self._resample(
                audio_float,
                input_sample_rate,
                output_sample_rate
            )
            metadata['resampled'] = True
        else:
            metadata['resampled'] = False
        
        # Step 6: Convert to int16 PCM
        audio_int16 = self._to_int16(audio_float)
        
        # Step 7: Measure output level
        output_level = np.max(np.abs(audio_int16)) if len(audio_int16) > 0 else 0
        metadata['output_level'] = output_level
        
        # Log periodically
        if self.input_frame_count <= 10 or self.input_frame_count % 100 == 0:
            logger.debug(
                f"[AUDIO→GEMINI] Frame {self.input_frame_count}: "
                f"in_level={input_level:.0f}, "
                f"gain={metadata['gain_applied']:.1f}x, "
                f"out_level={output_level}, "
                f"{input_sample_rate}Hz→{output_sample_rate}Hz"
            )
        
        return audio_int16.tobytes(), metadata
    
    def process_gemini_to_webrtc(
        self,
        audio_bytes: bytes,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 48000
    ) -> Tuple[bytes, dict]:
        """
        Process audio from Gemini Live for WebRTC (WhatsApp)
        
        Args:
            audio_bytes: PCM audio bytes from Gemini
            input_sample_rate: Gemini's output rate (24kHz)
            output_sample_rate: WhatsApp's expected rate (48kHz)
            
        Returns:
            Tuple of (PCM bytes, metadata dict)
        """
        self.output_frame_count += 1
        
        metadata = {
            'frame_num': self.output_frame_count,
            'input_bytes': len(audio_bytes),
        }
        
        # Convert bytes to int16 array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Measure input level
        input_level = np.max(np.abs(audio_int16)) if len(audio_int16) > 0 else 0
        metadata['input_level'] = input_level
        
        # Convert to float for resampling
        audio_float = audio_int16.astype(np.float32)
        
        # Resample to target rate
        if input_sample_rate != output_sample_rate:
            audio_float = self._resample(
                audio_float,
                input_sample_rate,
                output_sample_rate
            )
            metadata['resampled'] = True
        else:
            metadata['resampled'] = False
        
        # Convert back to int16
        audio_output = self._to_int16(audio_float)
        
        # Measure output
        output_level = np.max(np.abs(audio_output)) if len(audio_output) > 0 else 0
        metadata['output_level'] = output_level
        metadata['output_bytes'] = len(audio_output) * 2
        
        # Log periodically
        if self.output_frame_count <= 10 or self.output_frame_count % 100 == 0:
            logger.debug(
                f"[GEMINI→AUDIO] Frame {self.output_frame_count}: "
                f"{len(audio_bytes)}b in, {len(audio_output)*2}b out, "
                f"{input_sample_rate}Hz→{output_sample_rate}Hz"
            )
        
        return audio_output.tobytes(), metadata
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono if it's multi-channel"""
        if len(audio.shape) == 1:
            # Already mono
            return audio
        elif len(audio.shape) == 2:
            if audio.shape[0] == 1:
                # Single channel in planar format (1, N)
                return audio[0]
            elif audio.shape[0] == 2:
                # Stereo in planar format (2, N) - mix to mono
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 1:
                # Mono in interleaved format (N, 1)
                return audio[:, 0]
            elif audio.shape[1] == 2:
                # Stereo in interleaved format (N, 2) - mix to mono
                return np.mean(audio, axis=1)
            else:
                # Unknown layout - flatten
                logger.warning(f"Unknown audio shape: {audio.shape}, flattening")
                return audio.flatten()
        else:
            # Higher dimension - flatten
            logger.warning(f"Unexpected audio shape: {audio.shape}, flattening")
            return audio.flatten()
    
    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to float32 range suitable for processing
        
        Different formats need different handling:
        - int16: -32768 to 32767
        - int32: -2147483648 to 2147483647 (usually only using int16 range)
        - float32/float64: Usually -1.0 to 1.0
        """
        if audio.dtype == np.int16:
            # Standard int16 PCM
            return audio.astype(np.float32)
        
        elif audio.dtype == np.int32:
            # int32 audio - scale down to int16 range
            # WhatsApp sometimes uses int32 but only populates int16 range
            return (audio / 65536.0).astype(np.float32)
        
        elif audio.dtype in (np.float32, np.float64):
            # Float audio is typically normalized to [-1.0, 1.0]
            # Scale to int16 range for consistent processing
            return (audio * 32767.0).astype(np.float32)
        
        else:
            # Unknown format - try to convert
            logger.warning(f"Unknown audio dtype: {audio.dtype}, attempting conversion")
            return audio.astype(np.float32)
    
    def _apply_smart_gain(
        self,
        audio: np.ndarray,
        target_level: float = 8000.0,
        noise_threshold: float = 15.0,
        max_gain: float = 300.0
    ) -> Tuple[np.ndarray, float]:
        """
        Apply intelligent gain to quiet WhatsApp audio
        
        Args:
            audio: Float32 audio data
            target_level: Target peak level
            noise_threshold: Don't amplify if peak is below this (likely noise)
            max_gain: Maximum gain to prevent extreme amplification
            
        Returns:
            Tuple of (amplified audio, gain applied)
        """
        if len(audio) == 0:
            return audio, 1.0
        
        # Calculate peak level
        peak = np.max(np.abs(audio))
        
        # If audio is too quiet, it's probably just noise
        if peak <= noise_threshold:
            return audio, 1.0
        
        # Calculate required gain
        gain = target_level / peak
        
        # Cap gain to prevent distortion
        gain = min(gain, max_gain)
        
        # Apply gain
        audio_amplified = audio * gain
        
        return audio_amplified, gain
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) level of audio"""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def _resample(
        self,
        audio: np.ndarray,
        input_rate: int,
        output_rate: int
    ) -> np.ndarray:
        """
        Resample audio using linear interpolation
        
        This is a simple but effective resampling method.
        For production, could use scipy.signal.resample for better quality.
        """
        if input_rate == output_rate:
            return audio
        
        # Calculate number of output samples
        ratio = output_rate / input_rate
        num_output_samples = int(len(audio) * ratio)
        
        if num_output_samples == 0:
            return np.array([], dtype=audio.dtype)
        
        # Create interpolation indices
        indices = np.linspace(0, len(audio) - 1, num_output_samples)
        
        # Interpolate
        audio_resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return audio_resampled
    
    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 PCM, clipping to prevent overflow"""
        # Clip to int16 range
        audio_clipped = np.clip(audio, -32768, 32767)
        
        # Convert to int16
        return audio_clipped.astype(np.int16)
    
    def is_speech(
        self,
        audio_int16: np.ndarray,
        threshold: float = 500.0
    ) -> bool:
        """
        Simple voice activity detection
        
        Args:
            audio_int16: Int16 audio data
            threshold: Minimum peak level to consider as speech
            
        Returns:
            True if audio likely contains speech
        """
        if len(audio_int16) == 0:
            return False
        
        peak = np.max(np.abs(audio_int16))
        return peak > threshold


# Global instance for reuse
_pipeline = None


def get_audio_pipeline() -> AudioPipeline:
    """Get global audio pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = AudioPipeline()
    return _pipeline
