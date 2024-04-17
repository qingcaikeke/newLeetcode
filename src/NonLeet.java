import java.util.*;

/**
 * @author yjy
 * @date 2024/4/5
 * @Description
 */
public class NonLeet {
    public static void main(String[] args) {

    }
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

}
