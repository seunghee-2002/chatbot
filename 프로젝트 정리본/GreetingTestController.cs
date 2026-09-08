// Assets/_Projects/Scripts/UI/Controllers/GreetingTest/GreetingTestController.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;

namespace MagicRentalShop
{
    public class GreetingTestController : MonoBehaviour
    {
        [SerializeField] private GreetingTestView view;

        [Header("Ollama 설정")]
        [SerializeField] private string serverUrl = "http://192.168.0.6:11435/api/chat";

        // 모델 정의
        private static readonly string[] ModelNames = new[]
        {
            "qwen-2.5-1",
            "model_q4" // Llama-3.1-8B-Instruct
        };

        private int selectedModelIndex = 0;
        private string CurrentModel => ModelNames[selectedModelIndex];

        private readonly Queue<(GreetingTestParams p, GreetingResultCard card)> requestQueue
            = new Queue<(GreetingTestParams, GreetingResultCard)>();
        private bool isProcessing = false;

        private static readonly GreetingTestParams[] TestCases = new GreetingTestParams[]
        {
            new GreetingTestParams { personality="존칭형",  adventurerType="도적",   ageGroup="10대", gender="남성", grade="D급", visitCount="첫방문", lastWeapon="없음",   revisitGap="없음",   questResult="첫방문" },
            new GreetingTestParams { personality="야성형",  adventurerType="전사",   ageGroup="30대", gender="남성", grade="S급", visitCount="단골",   lastWeapon="도끼",   revisitGap="최근",   questResult="대성공" },
            new GreetingTestParams { personality="평범형",  adventurerType="마법사", ageGroup="20대", gender="여성", grade="B급", visitCount="보통",   lastWeapon="지팡이", revisitGap="보통",   questResult="실패"   },
            new GreetingTestParams { personality="너스레형", adventurerType="궁수",  ageGroup="20대", gender="여성", grade="C급", visitCount="적음",   lastWeapon="단검",   revisitGap="최근",   questResult="성공"   },
            new GreetingTestParams { personality="하대형",  adventurerType="전사",   ageGroup="30대", gender="남성", grade="A급", visitCount="많음",   lastWeapon="검",     revisitGap="오래됨", questResult="성공"   },
            new GreetingTestParams { personality="단답형",  adventurerType="도적",   ageGroup="20대", gender="남성", grade="A급", visitCount="보통",   lastWeapon="단검",   revisitGap="보통",   questResult="대성공" },
            new GreetingTestParams { personality="존칭형",  adventurerType="마법사", ageGroup="30대", gender="여성", grade="B급", visitCount="적음",   lastWeapon="망치",   revisitGap="보통",   questResult="실패"   },
            new GreetingTestParams { personality="평범형",  adventurerType="궁수",   ageGroup="20대", gender="여성", grade="C급", visitCount="많음",   lastWeapon="석궁",   revisitGap="오래됨", questResult="성공"   },
            new GreetingTestParams { personality="너스레형", adventurerType="전사",  ageGroup="10대", gender="남성", grade="A급", visitCount="보통",   lastWeapon="마법서", revisitGap="최근",   questResult="대성공" },
            new GreetingTestParams { personality="야성형",  adventurerType="도적",   ageGroup="20대", gender="남성", grade="B급", visitCount="보통",   lastWeapon="창",     revisitGap="보통",   questResult="실패"   },
        };

        #region 초기화

        private void Awake()
        {
            if (view == null)
                view = GetComponentInChildren<GreetingTestView>();
        }

        private void Start()
        {
            view?.SetController(this);
            view?.Initialize();
            view?.SetTestCaseButtonLabels(TestCases);
            view?.SetModelButtonLabels(ModelNames);
            view?.SetActiveModelButton(selectedModelIndex);
        }

        #endregion

        #region View로부터 호출되는 메서드

        public void OnRequestClicked()
        {
            Enqueue(view.GetCurrentParams());
        }

        public void OnTestCaseClicked(int index)
        {
            if (index < 0 || index >= TestCases.Length) return;
            view.SetDropdownsFromParams(TestCases[index]);
            Enqueue(TestCases[index]);
        }

        public void OnRunAllClicked()
        {
            foreach (var tc in TestCases)
                Enqueue(tc);
        }

        public void OnModelSelected(int index)
        {
            if (index < 0 || index >= ModelNames.Length) return;
            selectedModelIndex = index;
            view?.SetActiveModelButton(index);
            Debug.Log($"[GreetingTestController] 모델 변경: {CurrentModel}");
        }

        #endregion

        #region 큐 처리

        private void Enqueue(GreetingTestParams p)
        {
            GreetingResultCard card = view.AddResultCard(p, CurrentModel);
            requestQueue.Enqueue((p, card));
            UpdateStatus();

            if (!isProcessing)
                StartCoroutine(ProcessQueue());
        }

        private IEnumerator ProcessQueue()
        {
            isProcessing = true;

            while (requestQueue.Count > 0)
            {
                var (p, card) = requestQueue.Dequeue();
                UpdateStatus();
                yield return StartCoroutine(SendRequest(p, card));
            }

            isProcessing = false;
            view.SetStatus("완료");
        }

        private void UpdateStatus()
        {
            int queued = requestQueue.Count;
            int total = isProcessing ? queued + 1 : queued;
            if (total > 0)
                view.SetStatus($"처리 중... (대기 {queued}건)");
        }

        #endregion

        #region 내부 메서드

        private IEnumerator SendRequest(GreetingTestParams p, GreetingResultCard card)
        {
            string json = BuildJson(p);
            // 카드에 저장된 모델명으로 요청 (요청 시점의 선택 모델과 다를 수 있음)
            string model = card.ModelName;
            Debug.Log($"[GreetingTestController] 모델:{model} 요청: {json}");

            var body = new OllamaRequest
            {
                model    = model,
                messages = new List<OllamaMessage>
                {
                    new OllamaMessage { role = "user", content = json }
                },
                stream = false
            };

            byte[] raw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));

            using var req = new UnityWebRequest(serverUrl, "POST");
            req.uploadHandler   = new UploadHandlerRaw(raw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<OllamaResponse>(req.downloadHandler.text);
                string greeting = response?.message?.content ?? "(응답 없음)";
                card?.SetSuccess(greeting);
                Debug.Log($"[GreetingTestController] 결과: {greeting}");
            }
            else
            {
                card?.SetFailed(req.error);
                Debug.LogWarning($"[GreetingTestController] 실패: {req.error}");
            }

            UpdateStatus();
        }

        private string BuildJson(GreetingTestParams p)
        {
            return $@"{{
                ""성격"": ""{p.personality}"",
                ""모험가타입"": ""{p.adventurerType}"",
                ""나이"": ""{p.ageGroup}"",
                ""성별"": ""{p.gender}"",
                ""모험가등급"": ""{p.grade}"",
                ""방문횟수"": ""{p.visitCount}"",
                ""이전_아이템"": ""{p.lastWeapon}"",
                ""재방문간격"": ""{p.revisitGap}"",
                ""최근_의뢰"": ""{p.questResult}""
            }}";
        }

        #endregion

        #region Ollama 응답 구조

        [System.Serializable]
        private class OllamaResponse
        {
            public OllamaResponseMessage message;
        }

        [System.Serializable]
        private class OllamaResponseMessage
        {
            public string content;
        }

        #endregion
    }
}