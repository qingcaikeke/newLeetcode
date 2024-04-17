package todo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author yjy
 * @date 2024/4/12
 * @Description
 */
public class Main {
    public static void main(String[] args) {
        int[] arr = new int[0];
        System.out.println(Arrays.toString(arr));
    }
    public static int[] solution(int[] costs, int coins) {
        // write code here
        int[][] dp = new int[costs.length][coins+1];
        for(int j=1;j<=coins;j++){
            if(j>=costs[0]){
                dp[0][j] = 1;
            }
        }
        for(int i=1;i<costs.length;i++){//i代表当前能选择的硬币范围
            for(int j=1; j<=coins ;j++){//j代表当前能花的钱
                if(j>=costs[i]){//能买第i个物品
                    dp[i][j] = Math.max(dp[i-1][j] , dp[i-1][j-costs[i]]+1 );
                }else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        List<Integer> list = new ArrayList<>();
        int curCost = coins;
        for(int i = costs.length-1;i>0;i--){
            if(dp[i][curCost]>dp[i-1][curCost]){
                list.add(costs[i]);
                curCost -= costs[i];
                if(curCost<0) break;
            }
        }
        if(dp[0][curCost]>0){
            list.add(costs[0]);
        }
        int[] res = new int[list.size()];
        int index=0;
        for(int i=list.size()-1;i>=0;i--){
            res[index] = list.get(i);
            index++;
        }
        return res;
    }


//
//
//
//474 494 879
//    322 518 1449
//279 377 1049 1155






}