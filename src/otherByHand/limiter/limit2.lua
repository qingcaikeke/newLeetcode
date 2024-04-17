-- 定义令牌桶的容量和补充速率
local bucket_capacity = 100
local refill_rate = 10 -- 每秒补充10个令牌

-- 获取当前请求的IP和服务名
local ip = ngx.var.remote_addr
local service_name = "my_service"

-- 构造Redis键
local bucket_key = "bucket:" .. service_name .. ":" .. ip

-- 从Redis中获取令牌桶信息
local tokens, err = redis:get(bucket_key)
if not tokens then
    -- 如果令牌桶不存在,则初始化为满
    tokens = bucket_capacity
    redis:setex(bucket_key, 60, tokens) -- 设置1分钟过期时间
end

-- 如果令牌数足够,则消耗一个令牌
if tokens >= 1 then
    redis:decrby(bucket_key, 1)
    ngx.say("Request accepted")
else
    ngx.exit(503) -- 拒绝请求,返回503 Service Unavailable
end

-- 定期补充令牌
local ok, err = ngx.timer.at(1 / refill_rate, function(premature)
    if premature then
        return
    end
    local new_tokens, err = redis:eval([[
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens = tonumber(redis.call('get', key) or capacity)
        local new_tokens = math.min(tokens + refill_rate, capacity)
        redis.call('set', key, new_tokens)
        return new_tokens
    ]], 1, bucket_key, capacity, refill_rate)
    if err then
        ngx.log(ngx.ERR, "Error refilling tokens: ", err)
    end
end)