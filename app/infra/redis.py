"""Redis connectivity (Upstash rediss:// URL in production)."""

import redis

TIMEOUT_SECONDS = 2


def create_redis_client(url: str) -> redis.Redis:
    """Create a client without connecting; a rediss:// URL selects TLS."""
    return redis.Redis.from_url(url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS)


def check_redis(client: redis.Redis) -> None:
    """PING the server; raises redis-py's ConnectionError or TimeoutError on failure."""
    client.ping()
