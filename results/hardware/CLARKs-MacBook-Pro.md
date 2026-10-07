# Node: CLARKs-MacBook-Pro.local

| Field | Value |
|---|---|
| Chip | Apple M2 Pro |
| CPU cores | 10 |
| GPU cores | 16 |
| RAM | 16.0 GiB (17179869184 bytes) |
| GPU max recommended working set | 11.8 GiB (12713115648 bytes) |
| MLX default memory limit | 15.2 GiB (16320875724 bytes) |
| macOS | 27.2 |
| Python | 3.12.15 |
| mlx | 0.32.3 |
| MLX default device | Device(gpu, 0) |
| Thunderbolt / bridge IPs | {'bridge0': '169.254.141.243'} |
| RDMA devices (ibv_devices) | device          	   node GUID
    ------          	---------------- |
| Collected | 2026-10-07T08:18:09-0500 |

## mx.metal.device_info()

```json
{
  "device_name": "Apple M2 Pro",
  "max_recommended_working_set_size": 12713115648,
  "memory_size": 17179869184,
  "architecture": "applegpu_g14s",
  "max_buffer_length": 9534832640,
  "resource_limit": 499000
}
```

## Sanity check

```json
{
  "matmul_ok": true,
  "matmul_seconds": 0.0201,
  "matmul_result": 8589934592.0
}
```
