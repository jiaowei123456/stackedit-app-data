# Leetcode细节记录
### 1. 二分法：
获得比某个数x大的最小位次，已排序：
```Python
left = -1
right = len(nums)
while left + 1 < right:
	mid = (left+right)//2
	if nums[mid] < x:
		left = mid
	else:
		right = mid
return right 
```
### 2. 列表paixue：
获得比某个数x大的最小位次，已排序：
```Python
left = -1
right = len(nums)
while left + 1 < right:
	mid = (left+right)//2
	if nums[mid] < x:
		left = mid
	else:
		right = mid
return right 
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwMDY3OTAyMDJdfQ==
-->