--导包，使用 Nginx 提供的 resty.limit.req 库进行令牌桶限流
local limit_req = require "resty.limit.req"
local redis = require "resty.redis"

-- Redis 连接配置
local redis_host = "127.0.0.1"
local redis_port = 6379
local redis_timeout = 1000 -- 毫秒

-- 限流配置
local rate = 2 -- 每秒放入令牌的速率
local burst = 5 -- 令牌桶的容量

-- 获取客户端 IP 作为限流的 key
local key = ngx.var.binary_remote_addr

-- 连接 Redis
local red = redis:new()
red:set_timeout(redis_timeout)
local ok, err = red:connect(redis_host, redis_port)
if not ok then
    ngx.log(ngx.ERR, "failed to connect to Redis: ", err)
    return ngx.exit(500)
end

-- 获取当前令牌桶中的令牌数量
local tokens, err = red:get(key)
if not tokens then
    ngx.log(ngx.ERR, "failed to get tokens from Redis: ", err)
    return ngx.exit(500)
end

-- 如果不存在 key，或者令牌桶为空，则初始化令牌桶
if tokens == ngx.null then
    tokens = burst
    local ok, err = red:set(key, tokens, "EX", 1)
    if not ok then
        ngx.log(ngx.ERR, "failed to set tokens in Redis: ", err)
        return ngx.exit(500)
    end
end

-- 释放 Redis 连接
local ok, err = red:set_keepalive(10000, 100)
if not ok then
    ngx.log(ngx.ERR, "failed to set Redis keepalive: ", err)
end

-- 令牌桶限流
local lim, err = limit_req.new("rate_limit_store", rate, burst)
if not lim then
    ngx.log(ngx.ERR, "failed to instantiate a resty.limit.req object: ", err)
    return ngx.exit(500)
end

local delay, err = lim:incoming(key, true)
if not delay then
    if err == "rejected" then
        return ngx.exit(503)
    end
    ngx.log(ngx.ERR, "failed to limit req: ", err)
    return ngx.exit(500)
end

if delay >= 0.001 then
    ngx.sleep(delay)
end
