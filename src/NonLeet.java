import org.w3c.dom.ls.LSInput;

import java.util.*;

/**
 * @author yjy
 * @date 2024/4/5
 * @Description
 */
public class NonLeet {

    // 字节飞书一面
    // 给定一组进程的起始结束时间，求最大并行运行进程数
    //进程的起始时间的形式是[1,3]，1是开始时间，3是结束时间
    public static int maxProcess1(int[][] processes){
        TreeMap<Integer,Integer> treeMap = new TreeMap<>();
        //记录每个时间发生的事情
        for(int[] process:processes){
            treeMap.put(process[0], treeMap.getOrDefault(process[0], 0) + 1); // 开始时间加1
            treeMap.put(process[1], treeMap.getOrDefault(process[1], 0) - 1); // 结束时间减1
        }
        int curCount = 0;
        int res = 0;
        for (int countChange : treeMap.values()) {
            curCount += countChange;
            res = Math.max(res, curCount);
        }
        return res;
    }
    public static int maxProcess2(int[][] processes) {
        List<int[]> timeline = new ArrayList<>();
        // 将所有时间点和对应的开始/结束状态存储在timeline中
        for (int[] process : processes) {
            timeline.add(new int[]{process[0], 0}); // 开始时间，0表示开始
            timeline.add(new int[]{process[1], 1}); // 结束时间，1表示结束
        }
        // 根据时间对timeline进行排序
        Collections.sort(timeline, (a, b) -> a[0] - b[0]);
        int curCount = 0;
        int res = 0;
        // 遍历timeline，计算并发线程的最大数量
        for (int[] event : timeline) {
            if (event[1] == 0) {
                curCount++; // 开始线程
            } else {
                curCount--; // 结束线程
            }
            res = Math.max(res, curCount);
        }
        return res;
    }
    // 蚂蚁笔试 0413春招
    // 1.有每个物品的价格，n块钱最多买几样？ -- 贪心，价格排序，可便宜的买
    // 2.看+和*的矩阵，看里面*组成的字符时L还是T，可能翻转 -- T一定存在一个*周围有三个*
    // 3.一个数组，可以任意排序，排序后分成两部分，两部分的和为a和b，求a和b的最大公约数
    // 法1：每个元素有两种可能，选或不选，枚举2的n次方种组合，分别计算最大公约数，取最大的
    // 法2： x是a和b的公约数，那么a一定是a+b的约数，所以先找出a+b的所有约数，遍历约数，然后看能否从数组找出一个排列，是当前约数的倍数
    // 动态规划看是否存在一个子组，其和是x的倍数，
    public static int mayi_03(int[] nums){
        int sum = Arrays.stream(nums).sum();
        int n = nums.length;
        List<Integer> list = new ArrayList<>();
        for(int i=1;i<=sum/2;i++){
            if(sum%i==0){
                list.add(i);
            }
        }//答案一定再list种
        for(int i = list.size()-1;i>=0;i--){
            int cur = list.get(i);
            // 看能否从数组中找到一个组合是cur的倍数，能得话cur就是答案
            boolean[][] dp = new boolean[n+1][cur];
            dp[0][0] = true;
            //dp[i][j] 代表前i个元素能否找出一个子组，使子组和 取余 cur 余j
            for(int j =1;j<=n;j++){
                for(int k=0;k<cur;k++){
                    // 不选第j个元素，前j个能否凑出一个子组和取余cur余k 取决于 前j-1个能否凑出
                    dp[j][k] = dp[j-1][k] || dp[j][k];
                    // 选第j个，若前j-1凑和取余cur余k，那么前j个可以余 (num+k)%cur
                    if(dp[j-1][k]){
                        dp[j][(k+nums[j-1])%cur] = true;
                    }
                }
            }
            if(dp[n][0]){
                return cur;
            }
        }
        return 1;
    }
    // 辗转相除法求最大公约数
    public int gcd(int a,int b){
        int max = Math.max(a, b);
        int min = Math.min(a,b);
        if(max%min==0){
            return min;
        }
        return gcd(max%min,min);
    }
    // 腾讯音乐，可理解为一个数组中有多少种子组的和是奇数
    // 两行dp，dp[0][i] 前i个元素有多少种组合是偶数，dp[1][i] 前i个有多少种奇数
    // if(nums[i-1] 奇，dp[i] = dp[i-1][1] 不选 + dp[i-1][0] 选
    // if(nums[i-1] 偶，dp[i] = dp[i-1][1] + dp[i-1][1]


}
