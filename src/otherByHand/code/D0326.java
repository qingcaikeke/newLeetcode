package otherByHand.code;


import java.util.Random;

public class D0326 {
    public static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    public static class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
    //450.删除二叉搜索树指定节点
    public TreeNode deleteNode(TreeNode root,int val){
        //1.找到要删除的节点，如果节点只有左右子树中的一个，直接返回就行
        //2.否则，找到右子树的最左节点，把要删除节点替换，再删除右子树最左节点
        //返回删除之后的根节点
        if(root==null) return root;
        if(root.val==val){
            if(root.left==null) return root.right;
            if(root.right==null) return root.left;
            TreeNode node = root.right;//找右子树的最左节点
            while (node.left!=null){
                node = node.left;
            }
            root.right = deleteNode(root.right,node.val);//关键
            node.left = root.left;
            node.right = root.right;
            return node;
        } else if (root.val<val) {
            root.right = deleteNode(root.right,val);
        }else {
            root.left = deleteNode(root.left,val);
        }
        return root;
    }

    //找第k大的数,快排全是1会超时，最好在分割的过程返回三个list
    //建一个大小为k的小顶堆
    public static int getKth(int k,int[] nums,int left,int right){
        int index = partition(nums,left,right);
        int count = right-index+1;
        if(count==k) return nums[index];
        else if(count>k) return getKth(k,nums,index+1,right);
        else return getKth(k-count,nums,index,right);
    }
    public static int partition(int[] nums,int left,int right){
        int pivot = right;//Random random = new Random();int p = left+ random.nextInt(right-left+1);到不了1，所以最大是right
        int index = left-1;
        for(int i=left;i<=right;i++){
            if(nums[i]<=nums[pivot]){
                index++;
                swap(nums,index,i);
            }
        }
        return index;
    }




}
