/**
 * Voice RAG — microphone capture AudioWorklet (v5.0.0)
 * Collects mono Float32 samples at the AudioContext rate and posts them to the
 * main thread in fixed-size chunks; the main thread downsamples to 16 kHz PCM16
 * (RagUtils.downsampleTo16kPCM) and sends the frames over the WebSocket.
 * Replaces the deprecated ScriptProcessorNode (still used as a fallback in app.js).
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    const size = Number.isInteger(opts.chunkSize) && opts.chunkSize > 0 ? opts.chunkSize : 4096;
    this._buffer = new Float32Array(size);
    this._length = 0;
    this._active = true;
    this.port.onmessage = (event) => {
      if (event.data === 'stop') {
        this._active = false;
      }
    };
  }

  process(inputs) {
    if (!this._active) return false;
    const input = inputs[0];
    const channel = input && input[0];
    if (channel && channel.length) {
      let offset = 0;
      while (offset < channel.length) {
        const count = Math.min(channel.length - offset, this._buffer.length - this._length);
        this._buffer.set(channel.subarray(offset, offset + count), this._length);
        this._length += count;
        offset += count;
        if (this._length === this._buffer.length) {
          const chunk = this._buffer;
          this.port.postMessage(chunk, [chunk.buffer]);
          this._buffer = new Float32Array(chunk.length);
          this._length = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
