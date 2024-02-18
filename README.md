# git 명령어 모음

## git에 적용하지 않을 내용
.gitignore 파일을 이용하여 배제할 요소를 지정(주석은 #)

## 버전 관리
git add . 

git commit

### 과거로 되돌리기
git revert --continue

git reset --hard (해시)

git revert --no-commit (해시) : 리버트까지 다음번 커밋에 한번에 넣겠다

## 브랜치 만들기

- 생성 : git branch add-coach
 
- 목록확인 : git branch
 
- 이동 : git switch add-coach
  
- 생성과 동시에 이동 : git switch -c new-teams
  
- 삭제 : git branch -d (삭제할 브랜치명)

- 강제삭제 : git branch -D (강제삭제할 브랜치명)

- 이름바꾸기 : git branch -m (기존 브랜치명) (새 브랜치명)

- 여러개의 branch가 있는 경우 git log를 찍으면 HEAD가 위치한 부분에서 시작해서 아래로 쭉 타고 내려감

- 내역 편하게 보기 : git log --all --decorate --oneline --graph

# Readme.md 파일 작성법

