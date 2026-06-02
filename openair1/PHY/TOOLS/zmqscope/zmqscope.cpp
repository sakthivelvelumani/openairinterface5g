/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <thread>
#include <atomic>
#include "radio/zmq/ring_buffer.h"

#include <common/utils/LOG/log.h>

typedef struct __attribute__((packed)) {
  uint32_t buf_id;
  uint32_t nsamps;
} zmq_scope_hdr_t;

typedef struct {
  void *socket_;
  std::thread poll_thread;
  std::atomic<bool> poll_thread_running;
  overflow_buffer<uint8_t> buffer;
} zmq_scope_chan_t;

void enq_buf(zmq_scope_chan_t *s, uint32_t *buf_id, const c16_t *samples, size_t nsamps)
{
  if (nsamps == 0)
    return;

  zmq_scope_hdr_t h = {.buf_id = *buf_id, .nsamps = nsamps};

  size_t overflow = 0;
  overflow += s->buffer.push_samples((uint8_t *)&h, sizeof(h));
  overflow += s->buffer.push_samples((uint8_t *)samples, sizeof(*samples) * nsamps);
  if (overflow)
    LOG_W(PHY, "Overflow on ZMQ scope transfer\n");
}

extern "C" void *zmqscope_init()
{
}

extern "C" void zmqscope_delete(void *s)
{
}
